from __future__ import annotations
from typing import (
    List,
    Dict,
    Tuple,
    Optional,
    TextIO,
    Set,
    Callable,
    TypeVar,
    Generic,
    Union,
)
import numpy as np
import numpy.typing as npt
from Errors import *


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
        self.incomingedges: dict[State, float] = {}
        self.outgoingedges: dict[State, float] = {}

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


#################################################STATE


class State:
    def __init__(self, id: int, type: str, num_emissions: int) -> None:
        self.type: str = type
        self.id: int = id
        self.parents: Dict[State, float] = {}
        self.children: Dict[Union[State, int], float] = {}
        self.emissions: Dict[str, float] = {}
        self.within: Optional[Node] = None
        self.expLen: int = -1
        self.num_emissions: int = num_emissions

    def addEmissions(self, ems) -> None:
        self.emissions = {}

    def getExpLen(self) -> int:
        return self.expLen

    def setExpLen(self, len: int) -> None:
        self.expLen = len

    def addParent(self, parent: State) -> None:
        odds = parent.updateChild(self)
        self.parents[parent] = odds

    def updateChild(self, child: State) -> float:
        self.addChild(child, self.children[child.id])
        self.children.pop(child.id)
        return self.children[child]

    def addChildrenBulk(self, children: Dict[Union[State, int], float]):
        self.children = children
        # if self.children.get(self.id, None) is not None:
        #     self.children[self] = self.children[self.id]
        #     self.children.pop(self.id)

    def addChild(self, child: State, odds: float) -> None:
        self.children[child] = odds

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
        Within: {self.within.__repr__()}\n \
        ExpLen: {self.expLen}"

    def __repr__(self) -> str:
        return f"{self.type}({self.id})"
