"""
=============================================
Metacognitive accuracy calculation
=============================================

Loads sample decisions and confidence values, and returns the metacognitive
accuracy of each participant.

Author: Davide Valeriani

"""

from cBCI.individual.performance import metacognitive_accuracy
import pandas as pd
import numpy as np

df = pd.read_csv("data/decisions.csv", index_col=0, dtype=int)

correctness = df["correct"].values
df = df.drop(["correct"], axis=1)
decisions = df[df.columns[df.columns.to_series().str.contains('d')]].to_numpy().T
confidence = df[df.columns[df.columns.to_series().str.contains('c')]].to_numpy().T
confidence = (confidence - 1) / 3
confidence[confidence < 0] = np.nan

for s in range(decisions.shape[0]):
    print(metacognitive_accuracy(1, 0.3))
    print(metacognitive_accuracy((decisions[s] == correctness).astype(int), confidence[s]))
    break
