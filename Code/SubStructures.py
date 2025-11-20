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


class S(State):
    def __init__(self, id: int) -> None:
        super().__init__(id, "S", 0)


class E(State):
    def __init__(self, id: int) -> None:
        super().__init__(id, "E", 0)


class NONE(State):
    def __init__(self) -> None:
        super().__init__(-1, "NONE", 0)
