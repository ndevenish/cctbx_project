#!/usr/bin/env python3

"""
Finds classes with a show() but no __str__, and adds the method.

Requires the 'bowler' python library.

Usage:
    $ bowler run fix_show.py [--do]

Options:
    --do    Rewrite the destination files
"""

import sys

from bowler import Query
from bowler.types import LN, Capture, Filename

from fissix.pygram import python_symbols
from fissix.pytree import type_repr, Node, Leaf
from fissix.pgen2 import token
from fissix.fixer_util import Name, LParen, RParen, find_indentation, touch_import

from typing import Optional, List


import fissix.pygram
import fissix.pgen2
import fissix.pytree

# Build a driver to help generate nodes from known code
driver = fissix.pgen2.driver.Driver(
    fissix.pygram.python_grammar, convert=fissix.pytree.convert
)


def print_node(node: LN, max_depth: int = 1000, indent: str = "", last: bool = True):
    """Debugging function to print node tree"""
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


def get_children(node: Node, childtype: int) -> List[LN]:
    """Extract all children from a node that match a type"""
    return [x for x in node.children if x.type == childtype]


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


def process_class(node: LN, capture: Capture, filename: Filename) -> Optional[LN]:
    """Do the processing/modification of the class node"""
    print("show(): {}:{}".format(filename, node.get_lineno()))
    suite = get_child(node, python_symbols.suite)

    # Get the suite indent
    indent = find_indentation(suite)

    # To avoid having to work out indent correction, just generate with correct
    kludge_node = get_child(
        driver.parse_string(
            "def __str__(self):\n{0}{0}return kludge_show_to_str(self)\n\n".format(
                indent
            )
        ),
        python_symbols.funcdef,
    )

    # Work out if we have any trailing text/comments that need to be moved
    trail_node = get_trailing_text_node(suite)
    post = trail_node.prefix

    # if "maptbx" in filename and node.children[1].value == "spherical_variance_around_point":
    #     breakpoint()

    # The contents of this node will be moved
    trail_node.prefix = ""

    # Get the dedent node at the end of the previous - suite always ends with dedent
    # This is the dedent before the end of the suite, so the one to alter for the new
    # function
    # If we aren't after a suite then we don't have a DEDENT so don't need to
    # correct the indentation.
    #
    # children[-2] is the last statement at the end of the suite
    # children[-1] is the suite on a function definition
    # children[-1] is the dedent at the end of the function's suite
    if suite.children[-2].type == python_symbols.funcdef:
        last_func_dedent_node = suite.children[-2].children[-1].children[-1]
        last_func_dedent_node.prefix = "\n" + indent

    suite.children.insert(-1, kludge_node)

    # Get the kludge dedent - now the last dedent
    kludge_dedent = kludge_node.children[-1].children[-1]
    kludge_dedent.prefix = post
    touch_import("libtbx.utils", "kludge_show_to_str", node)


def do_filter(node: LN, capture: Capture, filename: Filename) -> bool:
    """Filter out potential matches that don't qualify"""
    if "table_utils.py" in filename:
        return True

    suite = get_child(node, python_symbols.suite)
    func_names = [
        x.children[1].value for x in get_children(suite, python_symbols.funcdef)
    ]

    # If we already have a __str__ method, then skip this
    if "__str__" in func_names:
        print("__str__ show(): {}:{}".format(filename, node.get_lineno()))
        return False

    if "__repr__" in func_names:
        print("__repr__ show(): {}:{}".format(filename, node.get_lineno()))
        return False

    # If we don't inherit from object directly we could already have a __str__ inherited
    class_parents = []
    if len(node.children) == 7:
        if node.children[3].type == python_symbols.arglist:
            class_parents = get_child(node, python_symbols.arglist).children
        elif node.children[3].type == token.NAME:
            class_parents = [node.children[3]]
        else:
            raise RuntimeError("Unexpected node type")
    if class_parents:
        parent_list = {
            str(x).strip() for x in class_parents if not x.type == token.COMMA
        }
        if not parent_list == {"object"}:
            print(
                "classbase show(): {}:{} ({})".format(
                    filename, node.get_lineno(), parent_list
                )
            )
            return False

    return True


PATTERN = """
classdef< 'class' any+ ':'
              suite< any*
                     funcdef< 'def' name='show' any+ >
                     any* > >
"""


def main():
    do_write = "--do" in sys.argv
    (
        Query()
        .select(PATTERN)
        .filter(do_filter)
        .modify(process_class)
        .execute(interactive=False, write=do_write, silent=False)
    )
