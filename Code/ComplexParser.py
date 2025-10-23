from __future__ import annotations
from typing import List, Dict, Tuple, Optional, TextIO, Set, Callable, TypeVar, Generic
from graphviz import Digraph


class State:
    def __init__(self, id: int, type: str) -> None:
        self.type: str = type
        self.id: int = id
        self.parents: Set[State] = set()
        self.children: Set[State] = set()
        self.within: Optional[Node] = None

    def addParent(self, parents: State) -> None:
        self.parents.add(parents)

    def addChildren(self, children: State) -> None:
        self.children.add(children)

    def addNode(self, node: Node) -> None:
        self.within = node

    def __hash__(self) -> int:
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
        Within: {self.within.__repr__()}"

    def __repr__(self) -> str:
        return f"{self.type}({self.id})"


class MP(State):
    def __init__(self, id: int) -> None:
        super().__init__(id, "MP")


class ML(State):
    def __init__(self, id: int) -> None:
        super().__init__(id, "ML")


class MR(State):
    def __init__(self, id: int) -> None:
        super().__init__(id, "MR")


class IL(State):
    def __init__(self, id: int) -> None:
        super().__init__(id, "IL")


class IR(State):
    def __init__(self, id: int) -> None:
        super().__init__(id, "IR")


class D(State):
    def __init__(self, id: int) -> None:
        super().__init__(id, "D")


class B(State):
    def __init__(self, id: int) -> None:
        super().__init__(id, "B")


class S(State):
    def __init__(self, id: int) -> None:
        super().__init__(id, "S")


class E(State):
    def __init__(self, id: int) -> None:
        super().__init__(id, "E")


class NONE(State):
    def __init__(self) -> None:
        super().__init__(-1, "NONE")


class StructureError(Exception):
    """Custom exception raised for structural issues in the graph."""

    def __init__(self, message: str = "A structural graph error has occurred ") -> None:
        super().__init__(message)


class LexError(Exception):
    """Custom exception raised for structural issues in the graph."""

    def __init__(self, message: str = "A structural graph error has occurred ") -> None:
        super().__init__(message)


class Node:
    def __init__(self, id: int, type: str, num_states) -> None:
        self.id: int = id
        self.type: str = type
        self.num_state: int = num_states
        self.states: Set[State] = set()
        self.parents: Set[Node] = set()
        self.children: Set[Node] = set()

    def addState(self, state: State) -> None:
        if len(self.states) >= self.num_state:
            raise StructureError(
                f"Too Many States in Node: {self.id}, expected: {self.num_state}"
            )
        self.states.add(state)

    def addParent(self, parent: Node) -> None:
        self.parents.add(parent)

    def addChild(self, child: Node) -> None:
        self.children.add(child)

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
        Contains: {[i.__repr__() for i in self.states]}"

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
        super().__init__(id, "BEGR", 1)


class END(Node):
    def __init__(self, id: int):
        super().__init__(id, "END", 1)


class BEGLR(Node):
    def __init__(self, id: int):
        super().__init__(id, "BEGLR", 2)


class MATP(Node):
    def __init__(self, id: int):
        super().__init__(id, "MATP", 6)


def lexer(file: str) -> Node:
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
                        raise LexError(f"Un-defined node type: {type} for node: #{id}")
        # done lexing file

        for node in nodelist:
            # linking nodes nodes are not listed in order
            for state in node.states:
                for child in state.children:
                    if child.within != node:
                        node.addChild(child.within)
                for parent in state.parents:
                    if parent.within != node:
                        node.addParent(parent.within)

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

        match type:
            case "MP":
                state = MP(id)
            case "ML":
                state = ML(id)
            case "MR":
                state = MR(id)
            case "IL":
                state = IL(id)
            case "IR":
                state = IR(id)
            case "D":
                state = D(id)
            case "B":
                state = B(id)
            case "S":
                state = S(id)
            case "E":
                state = E(id)
            case _:
                raise LexError(f"Un-defined state type: #{type} for node: #{id}")

        state.addNode(node)
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

        node.addState(state)

    return node


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
        self.currentNode = self.root
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


def render_graph(edges, supers, filename="graph", fmt="png"):
    dot = Digraph()
    for u, v in edges:
        dot.node(str(u))
        dot.node(str(v))
        dot.edge(str(u), str(v))

    for key, value in supers.items():
        with dot.subgraph(name=f"cluster_{key}") as c:
            c.attr(style="rounded", color="lightgrey", label=f"Node {key}")
            for i in list(set(value)):
                c.node(i)

    dot.render(filename, format=fmt, cleanup=True)


def main():
    import sys

    # just testing for now
    node = lexer(sys.argv[1])
    # testNodeTraverse(node)
    testStateTraverse(node)

    states = list(node.states)

    for i in node.children:
        states += list(i.states)

    states = list(set(states))
    edges = []
    nodesin = {}
    while states:
        k = states.pop()
        if k.within is not None:
            nodesin[str(k.within.id)] = nodesin.get(str(k.within.id), []) + [str(k.id)]

        for ch in k.children:
            edges.append((str(k.id), str(ch.id)))

            if ch.within is not None:
                nodesin[str(ch.within.id)] = nodesin.get(str(ch.within.id), []) + [
                    str(ch.id)
                ]

    edges = list(set(edges))

    render_graph(edges, nodesin)


if __name__ == "__main__":
    main()
