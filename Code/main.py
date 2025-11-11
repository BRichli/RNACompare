import csv
from itertools import product
from pathlib import Path

from tqdm.contrib.concurrent import process_map
from zss import simple_distance

import ComplexParser
from ComplexParser import Node
import zss_alg


type Model = tuple[str, str, Node]


def f(models: tuple[Model, Model]):
    try:
        return (
            models[0][0],  # clan 1
            models[0][1],  # family 1
            models[1][0],  # clan 2
            models[1][1],  # family 2
            simple_distance(
                models[0][2],
                models[1][2],
                Node.get_children,
                Node.getSelf,
                zss_alg.id_dist_expLen,
            ),
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

        results = process_map(f, product(models, models))
        resultswriter.writerows(results)


if __name__ == "__main__":
    main()
