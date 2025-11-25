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

import removeSilent
from hmmlearn import hmm


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
        self.emission_matrix: npt.NDArray[np.float64] = np.array([0])
        self.emitted_strings = []
        self.model = None

    def extract_emission_matrix(self):
        self.emission_matrix = np.array(
            [e for e in map(lambda x: list(x.emissions.values()), sorted(self.states))]
        )

    def normalize(self) -> None:
        self.states = sorted(list(set(self.states)))
        self.parents = sorted(list(set(self.parents)))
        self.children = sorted(list(set(self.children)))

    def make_model(self):
        if self.model is not None:
            return

        emissions = np.vstack(
            (
                np.array([0.0 for _ in range(20)] + [1.0]),
                self.emission_matrix,
                np.array([0.0 for _ in range(20)] + [1.0]),
            )
        )
        transitions, emissions = removeSilent.normalize_and_remove(
            self.transition_matrix, emissions
        )
        size = np.shape(transitions)[0]
        model = hmm.CategoricalHMM(n_components=len(transitions[0, :]))
        model.startprob_ = np.array([1.0] + [0.0 for _ in range(size - 1)])
        model.transmat_ = transitions
        model.emissionprob_ = emissions
        self.model = model

    def emission_prob(self, string):
        return self.model.score(string)

    def generate_strings(self, num_string):
        if len(self.emitted_strings) >= num_string:
            return
        numtomake = num_string - len(self.emitted_strings)
        for i in range(numtomake):
            magic_dummy = 20
            symbol_string = [magic_dummy]
            symbol = ""
            state = np.random.choice(self.model.n_components, p=self.model.startprob_)
            while symbol != magic_dummy:
                state = np.random.choice(
                    self.model.n_components, p=self.model.transmat_[state]
                )
                symbol = np.random.choice(
                    self.model.emissionprob_.shape[1], p=self.model.emissionprob_[state]
                )
                if symbol != magic_dummy:
                    symbol_string.append(symbol)
            symbol_string.append(20)
            symbol_string = np.array(symbol_string)
            symbol_string = symbol_string.reshape(1, -1)
            self.emitted_strings.append(
                (symbol_string, self.emission_prob(symbol_string))
            )

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
        State.addEmissions(self, {})

    def addEmissions(self, ems) -> None:
        l1 = [
            "AA",
            "AC",
            "AG",
            "AU",
            "CA",
            "CC",
            "CG",
            "CU",
            "GA",
            "GC",
            "GG",
            "GU",
            "UA",
            "UC",
            "UG",
            "UU",
            "A",
            "C",
            "G",
            "U",
            "DUMMY",
        ]
        self.emissions = {rna: 0.0 for rna in l1}

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
