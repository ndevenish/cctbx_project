#!/usr/bin/env python3

"""
Finds classes with a show() but no __str__, and adds the method.

Requires the 'bowler' python library.

Usage:
    $ bowler run fix_show.py [-- [--do] [--silent]]

Options:
    --do    Rewrite the destination files
"""

import sys
import itertools

from bowler import Query
from bowler.types import LN, Capture, Filename

from fissix.pygram import python_symbols
from fissix.pytree import type_repr, Node, Leaf
from fissix.pgen2 import token

# import fissix.fixer_util as fixer_util
from fissix.fixer_util import (
    Name,
    LParen,
    RParen,
    find_indentation,
    touch_import,
    find_binding,
)

from typing import Optional, List, Tuple


import fissix.pygram
import fissix.pgen2
import fissix.pytree

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
        print(indent + "└─...{} children".format(len(children)))
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
    """Extract all children from a node that match a type"""
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
    else:
        raise RuntimeError("Unknown print statement type?")


class UnableToTransformError(Exception):
    pass


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
    showID = "{}:{}".format(filename, node.get_lineno())
    # if "cctbx/crystal/__init__.py" in filename:
    #     breakpoint()
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
        print("PASS: {}: Writes to stdout only".format(showID))
    else:
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
                    "Unexpected extra required parameter: {} vs writes to {}".format(
                        without_default, output_name
                    )
                )
            print(
                "PASS: {}: Writes to required argument {}".format(
                    showID, without_default & {output_name}
                )
            )
        elif with_default:
            # We don't have any required parameters. Check that we pass the stream in as a keyword
            if not {output_name} <= with_default:
                raise UnableToTransformError(
                    "No function arguments match stream write destination"
                )
            print(
                "PASS: {}: Writes to keyword argument {} (of {})".format(
                    showID, output_name, with_default
                )
            )
        else:
            # breakpoint()
            assert len(params) <= 1
            raise UnableToTransformError("Doesn't write to stdout; but no arguments?")

    return output_name


def split_list_on(collection, condition):
    entries = []
    current_entry = []
    for i, item in enumerate(collection):
        if condition(item):
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


def _split_arglist(node: Node):
    """Splits a typedarglist into separate data"""
    if node.type == python_symbols.argument:
        return [_handle_argument(node)]

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
    # return entries


def _extract_argument(node):
    assert node.type == python_symbols.argument
    if node.children[1].type == token.EQUAL:
        return (node.children[0].value, True, node.children[2])
    else:
        return (str(node).strip(), False, None)


def _split_typedargslist(node: Node):
    """Splits a typedarglist into separate data"""
    assert node.type == python_symbols.typedargslist
    # entries = []
    # index = 0
    # current_entry = []
    # while index < len(node.children):
    #     part = node.children[index]
    #     if part.type == token.COMMA:
    #         entries.append(current_entry)
    #         current_entry = []
    #     else:
    #         current_entry.append(part)
    #     index += 1
    # if current_entry:
    #     entries.append(current_entry)
    entries = split_list_on(node.children, lambda x: x.type == token.COMMA)

    converted = []
    for entry in entries:
        if len(entry) == 1:
            converted.append((entry[0].value, False, None))
        elif len(entry) == 3:
            assert entry[1].type == token.EQUAL
            converted.append((entry[0].value, True, entry[2]))
        else:
            raise RuntimeError(
                "Don't understand typedarglist parameter {}".format(entry)
            )
    return converted
    # return entries


def _get_function_arguments(node):
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


# class FunctionParameters(object):
#     def __init__(self, node, parent=None):
#         assert node.type == python_symbols.parameters
#         self.node = node
#         self.parent = parent
#         # Parse the items in the parameter list to find out what they are

#     def __getitem__(self, name):
#         # if isinstance(name, int):
#         pass


#     def __contains__(self, name):
#         pass

# class Function(object):
#     """Represents a function definition"""

#     def __init__(self, node):
#         assert node.type == python_symbols.funcdef
#         self.node = node
#         self._parameters = FunctionParameters(node.children[2])

#     @property
#     def name(self):
#         return self.node.children[1].value
#     @name.setter
#     def name(self, value):
#         self.node.children[1].value = value

#     @property
#     def parameters(self):
#         return self._parameters
#         # return self.node.children[2]


def process_class(node: LN, capture: Capture, filename: Filename) -> Optional[LN]:
    """Do the processing/modification of the class node"""
    print("show(): {}:{}".format(filename, node.get_lineno()))
    print(str(node).strip())
    print_node(node.children[2])
    # if "euclidean_model_matching.py" in filename:
    #     print_node(node)
    #     breakpoint()

    # if "ptest.py" in filename:
    #     breakpoint()

    analyse_show(node, filename)
    print()
    return

    suite = get_child(node, python_symbols.suite)

    # Get the suite indent
    indent = find_indentation(suite)

    # To avoid having to work out indent correction, just generate with correct
    kludge_node = get_child(
        driver.parse_string(
            "def __str__(self):\n{0}  return kludge_show_to_str(self)\n\n".format(
                indent
            )
        ),
        python_symbols.funcdef,
    )

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
    # Make sure the function is available
    touch_import("libtbx.utils", "kludge_show_to_str", node)


# PATTERN = """
# classdef< 'class' any+ ':'
#               suite< any*
#                      funcdef< 'def' name='show' any+ >
#                      any* > >
# """
PATTERN = """
funcdef< 'def' name='show' any+ >
"""


def main():
    do_write = "--do" in sys.argv
    do_silent = "--silent" in sys.argv
    (
        Query()
        .select(PATTERN)
        # .filter(do_filter)
        .modify(process_class)
        .execute(interactive=False, write=do_write, silent=do_silent)
    )
