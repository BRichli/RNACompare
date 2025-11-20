from __future__ import annotations
from typing import List, Dict, Tuple, Optional, TextIO, Set, Callable, TypeVar, Generic
import numpy as np
import numpy.typing as npt
from Errors import *
from Structures import *


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


#################################################STATES


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

    def addEmissions(self, ems) -> None:
        l = [
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
        ]
        ems = np.exp(ems)
        s = sum(ems)
        ems = map(lambda x: x / s, ems)
        self.emissions = {rna: prob for rna, prob in zip(l, ems)}


class ML(State):
    def __init__(self, id: int) -> None:
        super().__init__(id, "ML", 4)

    def addEmissions(self, ems) -> None:
        l = ["A", "C", "G", "U"]
        ems = np.exp(ems)
        s = sum(ems)
        ems = map(lambda x: x / s, ems)
        self.emissions = {rna: prob for rna, prob in zip(l, ems)}


class MR(State):
    def __init__(self, id: int) -> None:
        super().__init__(id, "MR", 4)

    def addEmissions(self, ems) -> None:
        l = ["A", "C", "G", "U"]
        ems = np.exp(ems)
        s = sum(ems)
        ems = map(lambda x: x / s, ems)
        self.emissions = {rna: prob for rna, prob in zip(l, ems)}


class IL(State):
    def __init__(self, id: int) -> None:
        super().__init__(id, "IL", 4)

    def addEmissions(self, ems) -> None:
        l = ["A", "C", "G", "U"]
        ems = np.exp(ems)
        s = sum(ems)
        ems = map(lambda x: x / s, ems)
        self.emissions = {rna: prob for rna, prob in zip(l, ems)}


class IR(State):
    def __init__(self, id: int) -> None:
        super().__init__(id, "IR", 4)

    def addEmissions(self, ems) -> None:
        l = ["A", "C", "G", "U"]
        ems = np.exp(ems)
        s = sum(ems)
        ems = map(lambda x: x / s, ems)
        self.emissions = {rna: prob for rna, prob in zip(l, ems)}


class D(State):
    def __init__(self, id: int) -> None:
        super().__init__(id, "D", 0)


class B(State):
    def __init__(self, id: int) -> None:
        super().__init__(id, "B", 0)


class S(State):
    def __init__(self, id: int) -> None:
        super().__init__(id, "S", 0)


class E(State):
    def __init__(self, id: int) -> None:
        super().__init__(id, "E", 0)


class NONE(State):
    def __init__(self) -> None:
        super().__init__(-1, "NONE", 0)
