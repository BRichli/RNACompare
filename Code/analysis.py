import numpy as np
import pandas as pd
from skbio import DistanceMatrix
from skbio.stats.distance import permanova


df = pd.read_csv("./results3.csv")

samples = pd.unique(df[["family1", "family2"]].values.ravel())
samples = np.array(samples)

index_map = {s: i for i, s in enumerate(samples)}

n = len(samples)
dist_mat = np.zeros((n, n))

for _, row in df.iterrows():
    s1 = row["family1"]
    s2 = row["family2"]
    d = row["distance"]

    i = index_map[s1]
    j = index_map[s2]

    dist_mat[i, j] = d
    dist_mat[j, i] = d

dm = DistanceMatrix(dist_mat, ids=samples)  # type: ignore

group_map = {}

for _, row in df.iterrows():
    group_map[row["family1"]] = row["clan1"]
    group_map[row["family2"]] = row["clan2"]

metadata = pd.DataFrame({"group": [group_map[s] for s in samples]}, index=samples)

result = permanova(dm, metadata["group"])

print(dm)
print(metadata)
print(result)
