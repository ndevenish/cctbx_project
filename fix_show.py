#!/usr/bin/env python3

"""
Finds classes with a show() but no __str__, and adds the method.

Requires the 'bowler' python library.

Usage:
    $ bowler run fix_show.py [-- [--do] [--silent]]

Options:
    --do        Rewrite the destination files
    --silent    Don't print out diffs
"""

# Implementation:
# ##############
#
# Finding of prospective nodes is done at the bottom of the file using
# bowler's `Query` method.
#
# This calls the function `do_filter` for each prospective node, for
# early rejection. Anything that passes this test gets rewritten.
#
# The actual rewriting of the AST happens in `process_class`.

import itertools
import sys
from typing import Any, Callable, Collection, List, Optional, Tuple

import fissix.pgen2
import fissix.pygram
import fissix.pytree
from bowler import Query
from bowler.types import LN, Capture, Filename
from fissix.fixer_util import find_indentation, touch_import
from fissix.pgen2 import token
from fissix.pygram import python_symbols
from fissix.pytree import Leaf, Node, type_repr

# Build a driver to help generate nodes from known code
driver = fissix.pgen2.driver.Driver(
    fissix.pygram.python_grammar, convert=fissix.pytree.convert
)


def print_node(node: LN, max_depth: int = 1000, indent: str = "", last: bool = True):
    """Debugging function to print node tree.

    Arguments:
        node: The node to print
        max_depth: The maximum recursion depth to walk children
    """
    if last:
        first_i = "└─"
        second_i = "  "
    else:
        first_i = "├─"
        second_i = "│ "
    prefix = indent + first_i
    if type(node) is Node:
        print(
            prefix
            + "Node[{}] prefix={} suffix={}".format(
                type_repr(node.type), repr(node.prefix), repr(node.get_suffix())
            )
        )
    elif type(node) is Leaf:
        print(
            indent
            + first_i
            + "Leaf({}, {}, col={}{})".format(
                token.tok_name[node.type],
                repr(node.value),
                node.column,
                ", prefix={}".format(repr(node.prefix)) if node.prefix else "",
            )
        )
    else:
        raise RuntimeError("Unknown node type")
    indent = indent + second_i

    children = list(node.children)
    if max_depth == 0 and children:
        print(indent + f"└─...{len(children)} children")
    else:
        for i, child in enumerate(node.children):
            print_node(
                child,
                indent=indent,
                last=(i + 1) == len(children),
                max_depth=max_depth - 1,
            )


def get_child(node: Node, childtype: int) -> Optional[LN]:
    """Extract a single child from a node by type."""
    filt = [x for x in node.children if x.type == childtype]
    assert len(filt) <= 1
    return filt[0] if filt else None


def get_children(
    node: Node, childtype: int, *, recursive: bool = False, recurse_if_found=False
) -> List[LN]:
    """Extract all children from a node that match a type.

    Arguments:
        node: The node to search the children of. Won't be matched.
        childtype: The symbol/token code to search for
        recursive:
            If False, only the immediate children of the node will be searched
        recurse_if_found:
            If False, will stop recursing once a node has been found. If True,
            it is possible to have node types that are children of other nodes
            that were found earlier in the search.

    Returns:
        A list of nodes matching the search type.
    """
    if not recursive:
        return [x for x in node.children if x.type == childtype]
    matches = []
    for child in node.children:
        if child.type == childtype:
            matches.append(child)
            # If we want to stop recursing into found nodes
            if recurse_if_found:
                continue
        matches.extend(get_children(child, childtype, recursive=True))
    return matches


def get_print_file(node: Node) -> Optional[LN]:
    """Given a print node, get the output file.

    Returns:
        A node representing the name/destination, or None if not specified
    """
    if node.type == python_symbols.print_stmt:
        if node.children[1].type == token.RIGHTSHIFT:
            return node.children[2]
        return None
    elif node.type == python_symbols.power:
        # If not an argument list, then we know that we are sending to stdout
        argument = node.children[1].children[1]
        if argument.type == python_symbols.argument:
            arg_name, _, arg_val = _extract_argument(argument)
            if arg_name == "file":
                return arg_val
        elif argument.type == python_symbols.arglist:
            # Argument lists should be almost identical?
            args = _split_arglist(argument)
            for arg_name, _, arg_val in args:
                if arg_name == "file":
                    return arg_val
        else:
            return None

    raise RuntimeError("Unknown print statement type?")


class UnableToTransformError(Exception):
    """Thrown when a node doesn't qualify for transformation for some reason"""


def analyse_show(node: Node, filename: Filename) -> Optional[str]:
    """Analyse a show function to determine how to call it.

    Arguments:
        node: A funcdef Node

    Returns:
        The name of the keyword argument to pass file reader, or None if
        the function writes to stdout.

    Raises:
        UnableToTransformError: Function is too complicated to analyse
    """
    # Generate an ID for unique output
    showID = f"{filename}:{node.get_lineno()}"
    # Find calls to print in here
    # Firstly, print statements
    print_calls = get_children(node, python_symbols.print_stmt, recursive=True)
    # Now anything that looks like a print function
    print_calls.extend(
        [
            x
            for x in get_children(node, python_symbols.power, recursive=True)
            if str(x.children[0]).strip() == "print"
        ]
    )

    if not print_calls:
        raise UnableToTransformError("Could not find any output calls")

    # Get the node referring to every file that print writes to
    dest_file_nodes = [get_print_file(x) for x in print_calls]
    dest_file_vals = {
        str(x).strip() if x is not None else None for x in dest_file_nodes
    }
    print(
        "Output destinations:",
        ", ".join(str(x) if x is not None else "sys.stdout" for x in dest_file_vals),
    )
    # Require a single output destination for now.
    if not len(dest_file_vals) <= 1:
        raise UnableToTransformError(
            "Multiple output destinations: " + ", ".join(str(x) for x in dest_file_vals)
        )

    # The actual output stream used to output
    output_name = next(iter(dest_file_vals))

    # If this single output is 'None' then we print to stdout
    if output_name is None:
        print(f"PASS: {showID}: Writes to stdout only".format(showID))
    else:
        # Sort into required, optional function parameters
        params = _get_function_arguments(node)
        without_default = {
            name for name, _, _ in itertools.takewhile(lambda x: not x[1], params[1:])
        }
        with_default = {
            name for name, _, _ in itertools.takewhile(lambda x: x[1], params[1:])
        }

        if without_default:
            # We have required parameters. Check that it's just the log stream parameter
            if not without_default == {output_name}:
                raise UnableToTransformError(
                    f"Unexpected extra required parameter: {without_default} vs writes to {output_name}"
                )
            print(f"PASS: {showID}: Writes to required argument {output_name}")
        elif with_default:
            # We don't have any required parameters. Check that we pass the stream in as a keyword
            if not {output_name} <= with_default:
                raise UnableToTransformError(
                    "No function arguments match stream write destination"
                )
            print(
                f"PASS: {showID}: Writes to keyword argument {output_name} (of {with_default})"
            )
        else:
            assert len(params) <= 1
            raise UnableToTransformError("Doesn't write to stdout; but no arguments?")

    return output_name


def split_list_on(
    collection: Collection, predicate: Callable[[Any], bool]
) -> List[List]:
    """
    Splits a list into parts based on a predicate.

    This is the equivalent of str.split() but uses a function to
    determine if something is a split of not. Trailing split conditions
    will result in a trailing empty list.

    Arguments:
        collection: The input collection to split
        predicate:
            A callable function, called on each item, that returns True
            if the item should be used to split the list.

    Returns:
        A list with aa sublist for every part of the input that was
        divided by a splitter.
    """
    entries = []
    current_entry: List[Any] = []
    for i, item in enumerate(collection):
        if predicate(item):
            entries.append(current_entry)
            current_entry = []
            # If at the end, then we need to append an empty list
            if i == len(collection) - 1:
                entries.append([])
        else:
            current_entry.append(item)
    if current_entry:
        entries.append(current_entry)
    return entries


def test_split_list_on():
    assert split_list_on("123,43,5,", lambda x: x == ",") == [
        ["1", "2", "3"],
        ["4", "3"],
        ["5"],
        [],
    ]


def _split_arglist(node: Node) -> List[Tuple[str, bool, Optional[LN]]]:
    """Splits a arglist into separate data.

    Arguments:
        node: A python_symbols.argument or python_symbols.arglist node

    Returns:
        A list of separate parameters and information about it's default
        value, in the form (name, has_default, default)
    """
    if node.type == python_symbols.argument:
        return [_extract_argument(node)]

    assert node.type == python_symbols.arglist
    entries = split_list_on(node.children, lambda x: x.type == token.COMMA)

    converted = []
    for entry in entries:
        assert len(entry) == 1
        entry = entry[0]
        if entry.type == python_symbols.argument:
            converted.append(_extract_argument(entry))
        else:
            converted.append((str(entry).strip(), False, None))

    return converted


def _extract_argument(node: Node) -> Tuple[str, bool, Optional[LN]]:
    """Handle a single argument node.

    Arguments:
        node: A Node with type argument

    Returns:
        A list of separate parameters and information about it's default
        value, in the form (name, has_default, default)
    """
    assert node.type == python_symbols.argument
    if node.children[1].type == token.EQUAL:
        return (node.children[0].value, True, node.children[2])
    else:
        return (str(node).strip(), False, None)


def _split_typedargslist(node: Node) -> List[Tuple[str, bool, Optional[LN]]]:
    """Splits a typedargslist into separate data.

    Arguments:
        node: A python_symbols.typedargslist type Node

    Returns:
        A list of separate parameters and information about it's default
        value, in the form (name, has_default, default)
    """
    assert node.type == python_symbols.typedargslist
    entries = split_list_on(node.children, lambda x: x.type == token.COMMA)

    converted = []
    for entry in entries:
        if len(entry) == 1:
            converted.append((entry[0].value, False, None))
        elif len(entry) == 3:
            assert entry[1].type == token.EQUAL
            converted.append((entry[0].value, True, entry[2]))
        else:
            raise RuntimeError(f"Don't understand typedarglist parameter {entry}")
    return converted


def _get_function_arguments(node: Node) -> List[Tuple[str, bool, Optional[LN]]]:
    """
    Returns:
        (name, has_default, default)
    """
    if node.type == python_symbols.funcdef:
        node = node.children[2]
    else:
        assert node.type == python_symbols.parameters
    #
    if node.children[1].type == python_symbols.typedargslist:
        return _split_typedargslist(node.children[1])
    elif node.children[1].type == token.NAME:
        return [(node.children[1].value, False, None)]

    raise RuntimeError("Unknown configuration for python function parameters")


def get_trailing_text_node(node: Node) -> Leaf:
    """
    Find the dedent subnode containing any trailing text.

    If there are none, return the first.
    """
    # trails = []
    first = None
    for tail in reversed(list(node.leaves())):
        if not tail.type == token.DEDENT:
            break
        if first is None:
            first = tail
        if tail.prefix:
            return tail
    return first


def split_suffix(leaf: Leaf) -> Tuple[str, str]:
    """Split a suffix node (a DEDENT with prefix) into two.

    The indentation of the leaf is discovered so that comments that
    are inline with the previous block remain part of the prefix.
    """
    indent = find_indentation(leaf)
    parts = leaf.prefix.split("\n")
    pre = []
    for part in parts:
        if not part.startswith(indent):
            break
        pre.append(part)
    # Insert \n at the beginning of everything
    parts = [parts[0]] + ["\n" + x for x in parts[1:]]
    pre, post = "".join(parts[: len(pre)]), "".join(parts[len(pre) :])

    # If we have a pre but no newline, add/move one from post
    if pre and not pre.rstrip(" ").endswith("\n"):
        pre = pre + "\n"
        if post.startswith("\n"):
            post = post[1:]
    return pre, post


def process_class(node: LN, capture: Capture, filename: Filename) -> Optional[LN]:
    """Do the processing/modification of the class node"""
    print(f"Class for show(): {filename}:{node.get_lineno()}")

    suite = get_child(node, python_symbols.suite)
    # Get the suite indent
    indent = find_indentation(suite)

    # Find the show() function
    funcs = {
        x.children[1].value: x for x in get_children(suite, python_symbols.funcdef)
    }
    show_func = funcs["show"]
    # Get the name of the filename object keyword that this show uses
    show_keyword = analyse_show(show_func, filename)

    if show_keyword is None:
        # show() writes to stdout. Use Graeme's method for now.
        kludge_text = (
            f"def __str__(self):\n{indent}  return kludge_show_to_str(self)\n\n"
        )
    else:
        # We can more intelligently call show
        kludge_text = (
            f"def __str__(self):\n{indent}  out = StringIO()\n"
            f"{indent}  self.show({show_keyword}=out)\n"
            f"{indent}  return out.getvalue.rstrip()\n\n"
        )

    # To avoid having to work out indent correction, just generate with correct
    kludge_node = get_child(driver.parse_string(kludge_text), python_symbols.funcdef)

    # Work out if we have any trailing text/comments that need to be moved
    trail_node = get_trailing_text_node(suite)
    pre, post = split_suffix(trail_node)

    # The trailing contents of this node will be moved
    trail_node.prefix = pre

    # Get the dedent node at the end of the previous - suite always ends with dedent
    # This is the dedent before the end of the suite, so the one to alter for the new
    # function
    # If we aren't after a suite then we don't have a DEDENT so don't need to
    # correct the indentation.
    # children[-2] is the last statement at the end of the suite
    # children[-1] is the suite on a function definition
    # children[-1] is the dedent at the end of the function's suite
    if suite.children[-2].type == python_symbols.funcdef:
        last_func_dedent_node = suite.children[-2].children[-1].children[-1]
        last_func_dedent_node.prefix += "\n" + indent

    suite.children.insert(-1, kludge_node)

    # Get the kludge dedent - now the last dedent
    kludge_dedent = kludge_node.children[-1].children[-1]
    kludge_dedent.prefix = post

    # Make sure the functions used are available
    if show_keyword is None:
        touch_import("libtbx.utils", "kludge_show_to_str", node)
    else:
        touch_import("six.moves", "StringIO", node)


# Store a list of ids to ensure we don't process anything more than
# once. Normally we'd want this because bowler doesn't know that our
# changes aren't recursive, but we want to accurately count what's
# going on
_unique_attempts = set()


def do_filter(node: LN, capture: Capture, filename: Filename) -> bool:
    """Filter out potential matches that don't qualify"""

    # Make sure we don't process any class more than once
    classID = f"{filename}:{node.get_lineno()}"
    if classID in _unique_attempts:
        print("DUPLICATE_PARSE", classID)
        return False
    else:
        _unique_attempts.add(classID)

    # Print an explicit marker that we're starting
    print(f"FILTERING {classID}")
    suite = get_child(node, python_symbols.suite)
    func_names = [
        x.children[1].value for x in get_children(suite, python_symbols.funcdef)
    ]

    # If we already have a __str__ method, then skip this
    if "__str__" in func_names:
        print(f"ERROR:bowler.myfilter: __str__ show(): {classID}")
        return False

    if "__repr__" in func_names:
        print(f"ERROR:bowler.myfilter: __repr__ show(): {classID}")
        return False

    # If we don't inherit from object directly we could already have a __str__ inherited
    class_parents = []
    if len(node.children) == 7:
        if node.children[3].type == python_symbols.arglist:
            class_parents = get_child(node, python_symbols.arglist).children
        elif node.children[3].type in {token.NAME, python_symbols.power}:
            class_parents = [node.children[3]]
        else:
            raise RuntimeError(
                f"Unexpected node type in class argument: {type_repr(node.children[3].type)}"
            )

    if class_parents:
        parent_list = {
            str(x).strip() for x in class_parents if not x.type == token.COMMA
        }
        if not parent_list == {"object"}:
            print(
                f"ERROR:bowler.myfilter: Multiple non-object classbase show(): {classID} ({parent_list})"
            )
            return False

    print("FILTERING_PASS", classID)
    return True


PATTERN = """
classdef< 'class' any+ ':'
              suite< any*
                     funcdef< 'def' name='show' any+ >
                     any* > >
"""


def main():
    """Runs the query. Called by bowler if run as a script"""

    do_write = "--do" in sys.argv
    do_silent = "--silent" in sys.argv
    (
        Query()
        .select(PATTERN)
        .filter(do_filter)
        .modify(process_class)
        .execute(interactive=False, write=do_write, silent=do_silent)
    )


if __name__ == "__main__":
    main()
