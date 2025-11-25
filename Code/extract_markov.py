# extracts a truncated model from a node
from __future__ import annotations
from typing import List, Dict, Tuple, Optional, TextIO, Set, Callable, TypeVar, Generic
import ComplexParser as cp
import numpy as np
import numpy.typing as npt


def extractMatrix(node: Node) -> npt.NDArray[np.float64]:
    states = sorted(node.states)
    sids = [s.id for s in states]
    size = len(node.states) + 2

    indexmap = {states[a].id: a + 1 for a in range(len(states))}
    default_to = size + 2
    arr = np.full(
        (size + 2, size + 2), 0.0
    )  # initialize a 3d array where the first 2d slice represents the transition probabilities.

    for s in states:
        id = indexmap[s.id]

        for par, freq in s.transitions_from.items():
            parid = indexmap[par.id] if par.id in sids else 0
            arr[id, parid] = freq

        for ch, freq in s.transitions_to.items():
            chid = indexmap[ch.id] if ch.id in sids else default_to
            arr[id, chid] = freq

    # need to figure out transition emission probabilities

    return arr
