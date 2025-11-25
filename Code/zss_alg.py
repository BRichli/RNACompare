import zss
import ComplexParser
import sys


def id_dist_basic(a, b) -> int:
    """Uses node labels as proxy for node distance"""
    print(a)
    print(b)
    print(a.getSelf(a))
    if a.get_label(a) == b.get_label(b):
        return 0
    else:
        return 1


def id_dist_expLen(a, b) -> int:
    """Uses naive expected length of sequence as proxy for node distance"""
    if a.getId() == b.getId():
        return abs(a.getExpLen() - b.getExpLen())
    else:
        return 100


def insert_cost(a):
    return a.expLen


def remove_cost(a):
    return a.expLen


def update_cost(a, b):
    return abs(a.expLen - b.expLen)


def naive_size_dist(tree1, tree2):
    return zss.distance(
        tree1,
        tree2,
        ComplexParser.Node.get_children,
        insert_cost,
        remove_cost,
        update_cost,
    )


def main():
    tree1 = ComplexParser.lexer(sys.argv[1])
    tree2 = ComplexParser.lexer(sys.argv[2])

    # dist = zss.simple_distance(
    #     tree1,
    #     tree2,
    #     ComplexParser.Node.get_children,
    #     ComplexParser.Node.getSelf,
    #     id_dist_basic,
    # )

    dist = zss.distance(
        tree1,
        tree2,
        ComplexParser.Node.get_children,
        insert_cost,
        remove_cost,
        update_cost,
    )
    print("distance")
    print(dist)


if __name__ == "__main__":
    main()
