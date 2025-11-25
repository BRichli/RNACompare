from __future__ import annotations
from typing import List, Dict, Tuple, Optional, TextIO, Set, Callable, TypeVar, Generic
from graphviz import Digraph
from pathlib import Path
import numpy as np
import numpy.typing as npt
from Structures import *
from Errors import *
from SubStructures import *


# =======================TraversalMechanisms=================

T = TypeVar("T")
J = TypeVar("J")
M = TypeVar("M")


class NodeTraverse(Generic[T]):
    def __init__(
        self,
        visitorFn: Callable[[Node], T],
        *,
        prefix: Optional[bool] = True,
        root: Optional[Node] = None,
    ) -> None:
        self.visitorFn = visitorFn
        self.prefix = prefix
        self.root = root
        self.currentNode = None
        self.stack: List[Node] = []

    def set_graph(self, root: Node) -> None:
        self.root = root

    def __iter__(self) -> NodeTraverse:
        if self.root is None:
            raise ValueError("No Graph Supplied for Iteration")
        self.stack.append(self.root)
        return self

    def __next__(self) -> T:
        while (
            self.stack
        ):  # because this is a DAG we don't have to worry about infinite recursion
            self.currentNode = self.stack.pop(-1)  # prepare to look at next node
            self.stack += sorted(
                self.currentNode.children
            )  # put all of it's children on the stack
            return self.visitorFn(self.currentNode)  # process the next node.

        raise StopIteration


class StateTraverse(Generic[J]):
    def __init__(self, visitorFn: Callable[[State], J], stateList: List[State]) -> None:
        self.visitorFn = visitorFn
        self.stateList = sorted(stateList)

    def __iter__(self) -> StateTraverse:
        self.it = iter(self.stateList)
        return self

    def __next__(self) -> J:
        i = next(
            self.it
        )  # this is not an acyclic graph so we cant put anything else on the list to traverse.
        return self.visitorFn(i)


# =========================Tests


def testNodeTraverse(node) -> None:
    listofnodes = {}

    def nodeVisitor(n: Node) -> Dict[Node, str]:
        listofnodes[n] = n.kind
        return listofnodes

    traversal = NodeTraverse(nodeVisitor, root=node)
    traversal = traversal.__iter__()
    for x in traversal:
        pass

    print(listofnodes)


def testStateTraverse(node) -> None:
    listofStates = []

    def stateVisitor(s: State) -> List[Tuple[int, int]]:
        for i in s.children.keys():
            if isinstance(i, State):
                listofStates.append((s.id, i.id))
        return listofStates

    traversal = StateTraverse(stateVisitor, node.states)
    traversal = traversal.__iter__()

    for i in traversal:
        pass

    print(listofStates)


###=====================Graph Renderer==================
