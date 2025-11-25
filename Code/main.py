import csv
import math
from itertools import combinations, product
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from zss import simple_distance

import ComplexParser
import zss_alg
from ComplexParser import Node
from zss_alg import emission_distance, naive_size_dist

type Model = tuple[str, str, Node]


def insert_cost(a):
    return a.expLen


def remove_cost(a):
    return a.expLen


def update_cost(a, b):
    return abs(a.expLen - b.expLen)


def f(models: tuple[Model, Model]):
    try:
        return (
            models[0][0],  # clan 1
            models[0][1],  # family 1
            models[1][0],  # clan 2
            models[1][1],  # family 2
            # simple_distance(
            #     models[0][2],
            #     models[1][2],
            #     Node.get_children,
            #     Node.getSelf,
            #     zss_alg.id_dist_basic,
            # ),
            emission_distance(models[0][2], models[1][2]),
        )
    except Exception as e:
        e.add_note(f"{'/'.join(models[0][:2])} and {'/'.join(models[1][:2])}")
        raise


def main():
    cms = Path("../CMs").glob("CL*/*.cm")
    models = [(cm.parent.stem, cm.stem, ComplexParser.lexer(cm)) for cm in cms]

    with open("results.csv", "w", newline="") as resultsfile:
        resultswriter = csv.writer(resultsfile)
        resultswriter.writerow(["clan1", "family1", "clan2", "family2", "distance"])

        results = []
        for i in tqdm(combinations(models, 2), total=math.comb(len(models), 2)):
            try:
                results.append(f(i))
            except Exception as e:
                print("EXCEPTION")
                print(i[0][:2])
                print(i[1][:2])
                raise e
        resultswriter.writerows(results)


if __name__ == "__main__":
    main()
