#!/usr/bin/env python3

import sys
from bowler import Query
from fissix.pytree import type_repr, Node, Leaf
from fissix.pygram import python_symbols
from fissix.pgen2 import token  # token.COMMA

from fissix.refactor import RefactoringTool
from fissix.fixer_base import BaseFix

# from lib2to3.fixer_base import BaseFix
# from lib2to3.fixer_util import Leaf
# from lib2to3.pgen2 import token


# from fissix.pytree import type_repr, Node
def print_node(node, max_depth=1000, indent="", last=True):
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
    else:
        print(indent + first_i + repr(node))
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


def get_child(node, childtype):
    filt = [x for x in node.children if x.type == childtype]
    assert len(filt) <= 1
    return filt[0] if filt else None


def get_children(node, childtype):
    return [x for x in node.children if x.type == childtype]


# class ShowFixer(BaseFix):
#     PATTERN = """classdef<any*>"""

#     def transform(self, node, results):
#         print_node(node)


def test_modify(node, capture, filename):
    print_node(node)
    # breakpoint()
    # import pdb
    # pdb.set_trace()


def modify_show(node, capture, filename):
    if "postrefinement_legacy_rs.py" in filename:
        breakpoint()

    print("{}:{}".format(filename, node.get_lineno()))
    print_node(node,1)
    # breakpoint()
    if not node.type == python_symbols.funcdef:
        return
    # Step up to get the class
    classdef = node.parent.parent
    class_suite = node.parent
    if not node.parent.parent.type == python_symbols.classdef:
        if node.parent.parent.type == python_symbols.funcdef:
            print(
                "show() method at {}:{} belongs to function".format(
                    filename, node.get_lineno()
                )
            )
            return
        else:
            print("Unexpected node")
            breakpoint()
            return

    # Get a list of all functions
    # funcs = [x for x in classdef[-1].children if type_repr(x.type) == "funcdef"]

    # Look on the suite for functions
    # funcs = [x for x in classdef.children[-1].children if x.type == python_symbols.funcdef]
    func_names = [
        x.children[1].value for x in get_children(class_suite, python_symbols.funcdef)
    ]

    # If we already have a __str__ method, then skip this
    if "__str__" in func_names:
        print(
            "show() method at {}:{} belongs to class with existing __str__".format(
                filename, node.get_lineno()
            )
        )
        return

    if "__repr__" in func_names:
        print(
            "show() method at {}:{} belongs to class with existing __repr__".format(
                filename, node.get_lineno()
            )
        )
        return

    # If we don't inherit from object directly we could already have a __str__ inherited
    class_parents = get_child(classdef, python_symbols.arglist)
    if class_parents and not {
        str(x).strip() for x in class_parents.children if not x.type == token.COMMA
    } == {"object"}:
        print(
            "show() method for class {} ({}:{}) not directly from object".format(
                classdef.children[1].value, filename, classdef.get_lineno()
            )
        )
        return

    # We definitely have a show() function that belongs to a class without __str__
    def_def = "".join([str(x) for x in node.children[:-1]])
    print(
        "{}:{} {}::{}".format(
            filename,
            node.get_lineno(),
            classdef.children[1].value,
            "".join([str(x) for x in node.children[1:-1]]).lstrip(),
        )
    )
    # breakpoint()


class FixShowMethods(BaseFix):
    def match(self, node):
        print("DEBUG:Passed node ", node)
        return node.type == python_symbols.funcdef and node.children[1].value == "show"
        # if not node.type == python_symbols.funcdef:
        #     return False
        # breakpoint()

        # return is_funcdef and is_called_show

    def transform(self, node, results):
        print("DEBUG:    Parsing node", repr(node))
        breakpoint()


def apply(query):
    (
        # query.select_root()
        query.select_method("show")
        .modify(modify_show)
        .execute(silent=False, write=False)
    )


def main():
    """Run by bowler inline"""
    apply(Query())


# class ShowRefactorer(RefactoringTool):
#     def get_fixers(self):
#         fixers = [f(self.options, self.fixer_log) for f in self.fixers]
#         pre = [f for f in fixers if f.order == "pre"]
#         post = [f for f in fixers if f.order == "post"]
#         return pre, post

if __name__ == "__main__":
    path = sys.argv[1]
    apply(Query(path))

    # ref = ShowRefactorer([FixShowMethods])
    # filename = "tst_print.py"
    # with open(filename) as f:
    #     text = f.read()
    # breakpoint()
    # ref.refactor_string(text, filename)
