from __future__ import annotations
from typing import List, Dict, Tuple, Optional, TextIO, Set, Callable, TypeVar, Generic
from graphviz import Digraph
from pathlib import Path
import numpy as np
import numpy.typing as npt


def removeSilent(arr: npt.NDArray[np.float64], row: int) -> npt.NDArray[np.float64]:
    normd = normalize(arr[row], row)
    column = arr[:, row]
    adjustments = [normd * a for a in column]
    arrnew = adjustments + arr
    arrnew = np.delete(arrnew, row, axis=1)
    arrnew = np.delete(arrnew, row, axis=0)
    return arrnew


def normalize(items: npt.NDArray[np.float64], remove: int) -> npt.NDArray[np.float64]:
    s = sum(items) - items[remove]
    probs = items / s
    probs[-1] += 1.0 - probs.sum()
    return probs


def normalizeEms(items: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    ems = []
    for i in items:
        s = sum(i)
        ems.append(i / s)
    return np.array(ems)


def normalize_and_remove(tmatrix, ems):
    size = np.shape(tmatrix)[0]
    count = 1
    while count < size - 1:
        if sum(ems[count, :]) == 0:
            size = size - 1
            ems = np.delete(ems, count, axis=0)
            tmatrix = removeSilent(tmatrix, count)
        else:
            count += 1
    ems = normalizeEms(ems)

    return tmatrix, ems
