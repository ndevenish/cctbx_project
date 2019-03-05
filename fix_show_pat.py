from bowler import Query
from bowler.types import LN, Capture, Filename

from fissix.pygram import python_symbols
from fissix.pytree import type_repr, Node, Leaf
from fissix.pgen2 import token
from fissix.fixer_util import Name, LParen, RParen, find_indentation

from typing import Optional, List


import fissix.pygram
import fissix.pgen2
import fissix.pytree


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


# └─Node[funcdef] prefix='' suffix=''
#   ├─Leaf(1, 'def')
#   ├─Leaf(1, 'finalize_target_and_gradients')
#   ├─Node[parameters] prefix='' suffix=''
#   │ ├─Leaf(7, '(')
#   │ ├─Leaf(1, 'self')
#   │ └─Leaf(8, ')')
#   ├─Leaf(11, ':')
#   └─Node[suite] prefix='' suffix=''
#     ├─Leaf(4, '\n')
#     ├─Leaf(5, '    ')
#     ├─Node[simple_stmt] prefix='' suffix='    '
#     │ ├─Node[expr_stmt] prefix='' suffix=''
#     │ │ └─...3 children
#     │ └─Leaf(4, '\n')
#     ├─Node[if_stmt] prefix='    ' suffix=''
#     │ ├─Leaf(1, 'if')
#     │ ├─Node[atom] prefix=' ' suffix=''
#     │ │ └─...3 children
#     │ ├─Leaf(11, ':')
#     │ └─Node[suite] prefix='' suffix=''
#     │   └─...7 children
#     └─Leaf(6, '')


# def Colon():
#     """A colon"""
#     return Leaf(token.COLON, ":")

# def create_function():
#     func_children = [
#         Name("def"),
#         Name("show"),
#         Node(python_symbols.parameters, [LParen(), Name("self"), RParen()]),
#         Colon(),
#     ]
#     breakpoint()


# def indent(node, indent):
#     """Apply an extra indentation to a node tree"""
#     [x for x in lv if x.type == token.INDENT or x.type == token.DEDENT]

# def get_trailing_text(node):


def get_trailing_text_node(node):
    """
    Find the dedent node containing any trailing text.

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
    #   trails.insert(0, tail)
    # return "".join(x.prefix for x in trails)


def split_dedent_trails(prefix, indent):
    "Using the previous indent level, splits text pre-and-post indent"
    # parts = prefix.split("\n")
    # parts = [parts[0]] + ["\n" + x for x in parts[1:]]
    # # parts = [x + "\n" for x in parts[:-1]] + [parts[-1]]
    # pre = []
    # for i, part in enumerate(parts):
    #     if part.strip() and not (part.startswith(indent) or part.startswith("\n"+indent)):
    #         break
    # else:
    #     i += 1
    # return "".join(parts[:i]), "".join(parts[i:])
    parts = prefix.split("\n")
    return "\n".join(parts[:-1]), "\n"+parts[-1]
    # next_indent = parts[-1]
    # pre = ""
    # for i, part in enumerate(parts):
    #     if i == 0:
    #         pre += part
    #     else:
    #         pre += "\n" + part

def test_split_dedent_trails():
    assert split_dedent_trails("some", "  ") == ("", "some")
    assert split_dedent_trails("\nsome", "  ") == ("", "\nsome")
    assert split_dedent_trails("\n  some", "  ") == ("\n  some", "")
    assert split_dedent_trails("","  ") == ("","")


def process_class(node: LN, capture: Capture, filename: Filename) -> Optional[LN]:
    """Do the processing/modification of the class node"""
    print("show(): {}:{}".format(filename, node.get_lineno()))
    suite = get_child(node, python_symbols.suite)
    if "table_utils.py" in filename:
        breakpoint()
    # Get the suite indent
    indent = find_indentation(suite)

    kludge_node = get_child(
        driver.parse_string(
            "def __str__(self):\n{0}{0}return kludge_show_to_repr(self)\n\n".format(
                indent
            )
        ),
        python_symbols.funcdef,
    )

    # Work out if we have any trailing text/comments that need to be split
    trail_node = get_trailing_text_node(suite)
    # if trail_node:
    pre, post = split_dedent_trails(trail_node.prefix, indent)
    # trail_node.prefix = pre
    # else:
    #     pre, post = "",""

    # Get the dedent node at the end of the previous - suite always ends with dedent
    # children[-2] is the last statement at the end of the suite
    # children[-1] is the suite on a function definition
    # children[-1] is the dedent at the end of the function's suite
    # dedent_node = suite.children[-2].children[-1].children[-1]
    # if not dedent_node.prefix and trail_node:
    #     breakpoint()
    #     dedent_node = trail_node
    # old_dedent = dedent_node.prefix
    # assert dedent_node.type == token.DEDENT
    # dedent_node.prefix = "\n" + indent
    trail_node.prefix = pre+"\n"+indent
    suite.children.insert(-1, kludge_node)
    # Get the kludge dedent
    # breakpoint()
    kludge_dedent = kludge_node.children[-1].children[-1]
    kludge_dedent.prefix += post
    # breakpoint()

    # create_function()
    # breakpoint()


# print_node(driver.parse_string("class A(object):\n  def a(self):\n    pass\n\n  def b(self):\n    pass\n\npass\n"))


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
    (
        Query()
        .select(PATTERN)
        .filter(do_filter)
        .modify(process_class)
        .execute(interactive=False, write=False, silent=False)
    )

