import zss
import ComplexParser
from typing import List, Dict, Tuple, Optional, TextIO, Set, Callable, TypeVar, Generic
import sys


def id_dist(a, b) -> int:
    if a == b:
        return 0
    else:
        return 1


def main():
    tree1 = ComplexParser.lexer(sys.argv[1])
    tree2 = ComplexParser.lexer(sys.argv[2])

    dist = zss.simple_distance(
        tree1,
        tree2,
        ComplexParser.Node.get_children,
        ComplexParser.Node.get_label,
        id_dist,
    )

    print("distance")
    print(dist)


if __name__ == "__main__":
    main()
