from __future__ import annotations
from typing import List, Dict, Tuple, Optional, TextIO, Set, Callable, TypeVar, Generic
from graphviz import Digraph
from pathlib import Path
import numpy as np
import numpy.typing as npt


# ======================State Class


class State:
    def __init__(self, id: int, type: str, num_emissions: int) -> None:
        self.type: str = type
        self.id: int = id
        self.parents: List[State] = []
        self.children: List[State] = []
        self.transitions: Dict[int, float] = {}
        self.emissions: List[float] = []
        self.within: Optional[Node] = None
        self.expLen: int = -1
        self.num_emissions: int = num_emissions
        self.transitions_from: Dict[State, float] = {}
        self.transitions_to: Dict[State, float] = {}

    def normalize(self) -> None:
        self.parents = sorted(list(set(self.parents)))
        self.children = sorted(list(set(self.children)))

    def addTransitions(self, children: List[int], prob: List[float]) -> None:
        # print(f"State:{self.id} has transitions to :{children}, with prob: #{prob}")
        self.transitions = {c: p for c, p in zip(children, prob)}

    def addTransitionTo(self, child: State) -> None:
        self.transitions_to[child] = self.transitions[child.id]

    def addTransitionFrom(self, parent: State, prob: float) -> None:
        self.transitions_from[parent] = prob

    def addEmissions(self, ems) -> None:
        self.emissions = ems

    def getExpLen(self) -> int:
        return self.expLen

    def setExpLen(self, len: int) -> None:
        self.expLen = len

    def addParent(self, parents: State) -> None:
        self.parents.append(parents)

    def addChildren(self, children: State) -> None:
        self.children.append(children)

    def addNode(self, node: Node) -> None:
        self.within = node

    def __hash__(self) -> int:
        try:
            x = self.id
        except Exception as e:
            print(f"EXCEPTION HERE: {e}")
            raise e
        return self.id

    def __leq__(self, other: State) -> bool:
        return self.id <= other.id

    def __lt__(self, other: State) -> bool:
        return self.id < other.id

    def __eq__(self, other: object) -> bool:
        if isinstance(other, State):
            return self.id == other.id
        else:
            return False

    def __gt__(self, other: State) -> bool:
        return self.id > other.id

    def __gte__(self, other: State) -> bool:
        return self.id >= other.id

    def __str__(self) -> str:
        return f"State: {self.type}, ID: {self.id}, Parents: {[i.__repr__() for i in self.parents]}\n \
        Children: {[i.__repr__() for i in self.children]}\n \
        Within: {self.within.__repr__()}\n \
        ExpLen: {self.expLen}"

    def __repr__(self) -> str:
        return f"{self.type}({self.id})"


class DUMMYb(State):
    # A dummy begin state for truncating the HMM's within the Node
    def __init__(self) -> None:
        super().__init__(-3, "DUMMYB", 0)


class DUMMYe(State):
    # A dummy end state for truncating the HMM's within the Node
    def __init__(self) -> None:
        super().__init__(-2, "DUMMYe", 0)


class MP(State):
    def __init__(self, id: int) -> None:
        super().__init__(id, "MP", 16)


class ML(State):
    def __init__(self, id: int) -> None:
        super().__init__(id, "ML", 4)


class MR(State):
    def __init__(self, id: int) -> None:
        super().__init__(id, "MR", 4)


class IL(State):
    def __init__(self, id: int) -> None:
        super().__init__(id, "IL", 4)


class IR(State):
    def __init__(self, id: int) -> None:
        super().__init__(id, "IR", 4)


class D(State):
    def __init__(self, id: int) -> None:
        super().__init__(id, "D", 0)


class B(State):
    def __init__(self, id: int) -> None:
        super().__init__(id, "B", 0)

    def addTransitions(self, children: List[int], prob: List[float]) -> None:
        self.transitions[children[0]] = 0.0
        self.transitions[children[-1]] = 0.0


class S(State):
    def __init__(self, id: int) -> None:
        super().__init__(id, "S", 0)


class E(State):
    def __init__(self, id: int) -> None:
        super().__init__(id, "E", 0)


class NONE(State):
    def __init__(self) -> None:
        super().__init__(-1, "NONE", 0)


#### ==================Errors====================


class StructureError(Exception):
    """Custom exception raised for structural issues in the graph."""

    def __init__(self, message: str = "A structural graph error has occurred ") -> None:
        super().__init__(message)


class LexError(Exception):
    """Custom exception raised for structural issues in the graph."""

    def __init__(self, message: str = "A structural graph error has occurred ") -> None:
        super().__init__(message)


##===================Node Classes=================


class Node:
    @staticmethod
    def get_children(node: Node) -> List[Node]:
        return list(node.children)

    @staticmethod
    def get_label(node: Node) -> str:
        return node.type

    @staticmethod
    def getSelf(node: Node) -> Node:
        return node

    def __init__(self, id: int, type: str, num_states) -> None:
        self.id: int = id
        self.type: str = type
        self.num_state: int = num_states
        self.states: List[State] = []
        self.parents: List[Node] = []
        self.children: List[Node] = []
        self.expLen: int = -1
        self.transition_matrix: npt.NDArray[np.float64] = np.array([0])

    def normalize(self) -> None:
        self.states = sorted(list(set(self.states)))
        self.parents = sorted(list(set(self.parents)))
        self.children = sorted(list(set(self.children)))

    def getId(self) -> int:
        return self.id

    def getExpLen(self) -> int:
        return self.expLen

    def setExpLen(self, len: int) -> None:
        self.expLen = len

    def addState(self, state: State) -> None:
        if len(self.states) >= self.num_state:
            raise StructureError(
                f"Too Many States in Node: {self.id}, expected: {self.num_state}"
            )
        self.states.append(state)

    def addParent(self, parent: Node) -> None:
        self.parents.append(parent)

    def addChild(self, child: Node) -> None:
        self.children.append(child)

    def __hash__(self) -> int:
        return self.id

    def __leq__(self, other: Node) -> bool:
        return self.id <= other.id

    def __lt__(self, other: Node) -> bool:
        return self.id < other.id

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Node):
            return self.id == other.id
        else:
            return False

    def __gt__(self, other: Node) -> bool:
        return self.id > other.id

    def __gte__(self, other: Node) -> bool:
        return self.id >= other.id

    def __str__(self) -> str:
        return f"Node: {self.type}, ID: {self.id}, Parents: {[i.__repr__() for i in self.parents]}\n \
        Children: {[i.__repr__() for i in self.children]}\n \
        Contains: {[i.__repr__() for i in self.states]}\n \
        ExpLen: {self.expLen}"

    def __repr__(self) -> str:
        return f"{self.type}({self.id})"


class ROOT(Node):
    def __init__(self, id: int):
        super().__init__(id, "ROOT", 3)


class MATL(Node):
    def __init__(self, id: int):
        super().__init__(id, "MATL", 3)


class MATR(Node):
    def __init__(self, id: int):
        super().__init__(id, "MATR", 3)


class BIF(Node):
    def __init__(self, id: int):
        super().__init__(id, "BIF", 1)


class BEGL(Node):
    def __init__(self, id: int):
        super().__init__(id, "BEGL", 1)


class BEGR(Node):
    def __init__(self, id: int):
        super().__init__(id, "BEGR", 2)


class END(Node):
    def __init__(self, id: int):
        super().__init__(id, "END", 1)


class BEGLR(Node):
    def __init__(self, id: int):
        super().__init__(id, "BEGLR", 2)


class MATP(Node):
    def __init__(self, id: int):
        super().__init__(id, "MATP", 6)


# +++++++++++++++++++++Markov


def extractMatrix(node: Node) -> npt.NDArray[np.float64]:
    states = sorted(node.states)
    sids = [s.id for s in states]
    size = len(node.states)

    indexmap = {states[a].id: a + 1 for a in range(len(states))}
    default_to = size + 1
    arr = np.full(
        (size + 2, size + 2), 1.0
    )  # initialize a 3d array where the first 2d slice represents the transition probabilities.

    if node.id == 2:
        print("here")
    outmatrix = {x: 0.0 for x in indexmap.values()}
    inmatrix = {x: 0.0 for x in indexmap.values()}
    for s in states:
        id = indexmap[s.id]

        for par, freq in s.transitions_from.items():
            if par.id in sids:
                parid = indexmap[par.id]
                arr[id, parid] = freq
            else:
                inmatrix[id] += freq

        for ch, freq in s.transitions_to.items():
            if ch.id in sids:
                chid = indexmap[ch.id]
                arr[id, chid] = freq
            else:
                outmatrix[id] += freq

    for ind, val in outmatrix.items():
        arr[ind, default_to] = val
    for ind, val in inmatrix.items():
        arr[0, ind] = val
    arr[default_to, default_to] = 0.0
    # need to figure out transition emission probabilities

    return arr


# =====================Lexers


def lexer(file: str | Path) -> Node:
    with open(file, "r") as filestream:
        flag = True
        while flag:
            line = filestream.readline()
            if line.strip() == "CM":
                flag = False

        nodelist = []
        stateDict = {}
        flag = True
        while flag:
            line = filestream.readline().strip()
            if line == "//":
                flag = False
            else:
                tokens = [token for token in line.split() if token]

                type = tokens[1]
                id = int(tokens[2])

                match type:
                    case "ROOT":
                        nodelist.append(lexNode(ROOT(id), stateDict, filestream))
                    case "MATL":
                        nodelist.append(lexNode(MATL(id), stateDict, filestream))
                    case "MATR":
                        nodelist.append(lexNode(MATR(id), stateDict, filestream))
                    case "BIF":
                        nodelist.append(lexNode(BIF(id), stateDict, filestream))
                    case "BEGL":
                        nodelist.append(lexNode(BEGL(id), stateDict, filestream))
                    case "BEGR":
                        nodelist.append(lexNode(BEGR(id), stateDict, filestream))
                    case "END":
                        nodelist.append(lexNode(END(id), stateDict, filestream))
                    case "BEGLR":
                        nodelist.append(lexNode(BEGLR(id), stateDict, filestream))
                    case "MATP":
                        nodelist.append(lexNode(MATP(id), stateDict, filestream))
                    case _:
                        raise LexError(f"Un-defined node type: {type} for node: {id}")
        # done lexing file

        for node in nodelist:
            tempLen = 0
            # linking nodes nodes are not listed in order
            for state in node.states:
                tempLen += state.expLen
                for child in state.children:
                    if child.within != node:
                        node.addChild(child.within)
                for parent in state.parents:
                    if parent.within != node:
                        node.addParent(parent.within)
                state.normalize()

            node.setExpLen(tempLen)
            node.normalize()
            node.transition_matrix = extractMatrix(node)

        # at this point all states and all nodes should be linked.
    return nodelist[
        0
    ]  # we only need the start node everything else is linked at this point.


def lexNode(node: Node, stateDict: Dict[int, State], filestream: TextIO) -> Node:
    # adding states

    for _ in range(node.num_state):
        line = filestream.readline()
        tokens = [token for token in line.strip().split() if token]
        type = tokens[0]

        id = int(tokens[1])

        parents = []
        if int(tokens[2]) != -1:
            parents = list(
                range(int(tokens[2]) - int(tokens[3]) + 1, int(tokens[2]) + 1)
            )
        end_Transitions = int(tokens[5]) + 10 + 1  # plus one because not inclusive
        expLen = int(tokens[6])
        # starting from 10th get the transition probabilities
        transitions = [
            0.0 if x == "*" else float(x) for x in tokens[10:end_Transitions]
        ]
        children = list(range(int(tokens[4]), int(tokens[4]) + int(tokens[5]) + 1))

        match type:
            case "MP":
                state = MP(id)
                emissions = [float(x) for x in tokens[end_Transitions:]]
                state.addEmissions(emissions)

            case "ML":
                state = ML(id)
                emissions = [float(x) for x in tokens[end_Transitions:]]
                state.addEmissions(emissions)

            case "MR":
                state = MR(id)
                emissions = [float(x) for x in tokens[end_Transitions:]]
                state.addEmissions(emissions)

            case "IL":
                state = IL(id)
                emissions = [float(x) for x in tokens[end_Transitions:]]
                state.addEmissions(emissions)

            case "IR":
                state = IR(id)
                emissions = [float(x) for x in tokens[end_Transitions:]]
                state.addEmissions(emissions)

            case "D":
                state = D(id)
            case "B":
                state = B(id)
                children = [children[0], children[-1] - children[0]]
                transitions = [0.0, 0.0]
            case "S":
                state = S(id)
            case "E":
                state = E(id)
            case _:
                raise LexError(f"Un-defined state type: #{type} for node: #{id}")

        state.setExpLen(expLen)  # adding in the expected length
        state.addNode(node)

        state.addTransitions(children, transitions)

        stateDict[id] = state

        for item in parents:
            # linking states we can rely on the assumption that all states are listed in order. as specified by the infernal documentation
            try:
                state.addParent(stateDict[item])

            except KeyError as e:
                if e.args[0] == -1:
                    print(f"error: {e} for ")
                    print(id)
                    print(parents)
                    break
                else:
                    raise e

            stateDict[item].addChildren(state)
            stateDict[item].addTransitionTo(state)
            state.addTransitionFrom(
                stateDict[item], stateDict[item].transitions_to[state]
            )

        node.addState(state)

    return node


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
        listofnodes[n] = n.type
        return listofnodes

    traversal = NodeTraverse(nodeVisitor, root=node)
    traversal = traversal.__iter__()
    for x in traversal:
        pass

    print(listofnodes)


def testStateTraverse(node) -> None:
    listofStates = []

    def stateVisitor(s: State) -> List[Tuple[int, int]]:
        for i in s.children:
            listofStates.append((s.id, i.id))
        return listofStates

    traversal = StateTraverse(stateVisitor, node.states)
    traversal = traversal.__iter__()

    for i in traversal:
        pass

    print(listofStates)


###=====================Graph Renderer==================


def render_graph(edges, filename="graph", fmt="png"):
    dot = Digraph()
    for u, v in edges:
        dot.node(str(u))
        dot.node(str(v))
        dot.edge(str(u), str(v))

    # for key, value in supers.items():
    #     with dot.subgraph(name=f"cluster_{key}") as c:
    #         c.attr(style="rounded", color="lightgrey", label=f"Node {key}")
    #         for i in list(set(value)):
    #             c.node(i)

    dot.render(filename, format=fmt, cleanup=True)


# ========================Main


def main():
    import sys

    # just testing for now
    node = lexer(sys.argv[1])

    # edges = []

    def findnode(n):
        if n.id == 6:
            print(n.transition_matrix)

    traveler = NodeTraverse(findnode, root=node)

    for i in traveler:
        pass

    # testNodeTraverse(node)
    # testStateTraverse(node)

    # states = list(node.states)

    # for i in node.children:
    #     states += list(i.states)

    # states = list(set(states))
    # edges = []
    # nodesin = {}
    # while states:
    #     k = states.pop()
    #     if k.within is not None:
    #         nodesin[str(k.within.id)] = nodesin.get(str(k.within.id), []) + [str(k.id)]

    #     for ch in k.children:
    #         edges.append((str(k.id), str(ch.id)))

    #         if ch.within is not None:
    #             nodesin[str(ch.within.id)] = nodesin.get(str(ch.within.id), []) + [
    #                 str(ch.id)
    #             ]

    # edges = list(set(edges))

    # render_graph(edges)


if __name__ == "__main__":
    main()
