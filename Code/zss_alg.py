import zss
import ComplexParser
import sys


def id_dist_basic(a, b) -> int:
    """Uses node labels as proxy for node distance"""
    if a.get_label() == b.get_label():
        return 0
    else:
        return 1


def id_dist_expLen(a, b) -> int:
    """Uses naive expected length of sequence as proxy for node distance"""
    if a.getId() == b.getId():
        return abs(a.getExpLen() - b.getExpLen())
    else:
        return 100


def main():
    tree1 = ComplexParser.lexer(sys.argv[1])
    tree2 = ComplexParser.lexer(sys.argv[2])

    dist = zss.simple_distance(
        tree1,
        tree2,
        ComplexParser.Node.get_children,
        ComplexParser.Node.getSelf,
        id_dist_expLen,
    )

    print("distance")
    print(dist)


if __name__ == "__main__":
    main()
