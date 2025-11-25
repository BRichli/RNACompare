from __future__ import annotations
from typing import List, Dict, Tuple, Optional, TextIO, Set, Callable, TypeVar, Generic
from graphviz import Digraph
from pathlib import Path
import numpy as np
import numpy.typing as npt
from Structures import *
from Errors import *
from SubStructures import *
from TraversalMechanisms import *
from hmmlearn import hmm
import removeSilent


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
]


# +++++++++++++++++++++Markov
def normalize_logs(val: float, list: List[float]) -> np.float64:
    x = sum(map(np.exp, list))
    if x == 0.0:
        return np.float64(0.0)
    else:
        return np.float64(np.exp(val) / x)


def extractMatrix(node: Node) -> npt.NDArray[np.float64]:
    states = sorted(node.states)
    sids = [s.id for s in states]
    size = len(node.states)

    indexmap = {states[a].id: a + 1 for a in range(len(states))}
    default_to = size + 1
    arr = np.full(
        (size + 2, size + 2), 0.0
    )  # initialize a 3d array where the first 2d slice represents the transition probabilities.

    outmatrix = {x: np.float64(0.0) for x in indexmap.values()}
    inmatrix = {x: np.float64(0.0) for x in indexmap.values()}

    for s in states:
        id = indexmap[s.id]
        for par, freq in s.parents.items():
            if par.id in sids:
                parid = indexmap[par.id]
                arr[id, parid] = np.exp(freq)
            else:
                inmatrix[id] += normalize_logs(freq, list(par.children.values()))

        for ch, freq in s.children.items():
            if isinstance(ch, State) and ch.id in sids:
                chid = indexmap[ch.id]
                arr[id, chid] = np.exp(freq)
            else:
                outmatrix[id] += normalize_logs(freq, list(s.children.values()))

    for ind, val in outmatrix.items():
        arr[ind, default_to] = val
    for ind, val in inmatrix.items():
        arr[0, ind] = val
    arr[default_to, default_to] = 1.0

    # need to figure out transition emission probabilities
    if node.kind == "ROOT":
        # print("ROOT")
        # print(arr)
        arr = arr[1:, 1:]
        # print("new arr")
        # print(arr)

    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    arr = arr / row_sums
    # if node.kind == "ROOT":
    #     row_sums = arr.sum(axis=1, keepdims=True)
    #     print(row_sums)

    return arr


def logprobs_to_probs(probs):
    # Mask impossible transitions
    probs[probs == 1.0] = -np.inf  # sentinel

    # Exponentiate valid log-probs
    probs = np.exp(probs)

    # Reset impossible transitions to 0
    probs[np.isinf(probs)] = 0.0

    return probs


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
                for child in state.children.keys():
                    if child.within != node:
                        node.addChild(child.within)

                for parent in state.parents.keys():
                    if parent.within != node:
                        node.addParent(parent.within)

            node.setExpLen(tempLen)
            node.normalize()
            node.transition_matrix = extractMatrix(node)
            node.extract_emission_matrix()
            node.make_model()
            node.generate_strings(4)  # magic number

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
        children = list(range(int(tokens[4]), int(tokens[4]) + int(tokens[5])))

        match type:
            case "MP":
                state = MP(id)
                emissions = [float(x) for x in tokens[-16:]]
                state.addEmissions(emissions)

            case "ML":
                state = ML(id)
                emissions = [float(x) for x in tokens[-4:]]
                state.addEmissions(emissions)

            case "MR":
                state = MR(id)
                emissions = [float(x) for x in tokens[-4:]]
                state.addEmissions(emissions)

            case "IL":
                state = IL(id)

                emissions = [float(x) for x in tokens[-4:]]

                state.addEmissions(emissions)

            case "IR":
                state = IR(id)
                emissions = [float(x) for x in tokens[-4:]]
                state.addEmissions(emissions)

            case "D":
                state = D(id)
            case "B":
                state = B(id)
                children = [children[0], children[-1] - children[0] + 1]
                transitions = [0.0, 0.0]
            case "S":
                state = S(id)
            case "E":
                state = E(id)
            case _:
                raise LexError(f"Un-defined state type: #{type} for node: #{id}")

        state.setExpLen(expLen)  # adding in the expected length
        state.addNode(node)
        state.addChildrenBulk({ch: tr for ch, tr in zip(children, transitions)})

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

        node.addState(state)

    return node


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
    np.set_printoptions(precision=5, suppress=True)

    def findnode(n):
        if n.id == 6:
            return n

    x = node
    traveler = NodeTraverse(findnode, root=node)
    for i in traveler:
        if i is not None:
            x = i
            break

    # x.make_model()

    # x.generate_strings(4)

    for y in x.emitted_strings:
        print(y)

    # from pomegranate.hmm import DenseHMM, SparseHMM
    # from pomegranate.distributions import Categorical

    # distrs = []
    # for i in ems:
    #     distrs.append(Categorical([i]))
    # # model.add_distributions(distrs)

    # edges = []
    # for row_num, row in enumerate(tmatrix):
    #     for col_num, prob in enumerate(row):
    #         if prob > 0.0:
    #             edges.append([distrs[row_num], distrs[col_num], prob])

    # # model.add_edge(model.start, distrs[0], 1.0)
    # # model.add_edge(distrs[-1], model.end, 1.0)
    # starts = np.array([1.0] + [0.0 for _ in range(size - 1)])
    # ends = np.array([0.0 for _ in range(size - 1)] + [1.0])

    # model = SparseHMM(distributions=distrs, edges=edges, starts=starts, ends=ends)
    # model.add_distributions(distrs)
    # X_sample = model.sample(10)
    # print(X_sample)


if __name__ == "__main__":
    main()
