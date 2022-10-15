"""
=============================================
Collaborative BCI for target detection
=============================================

Loads sample decisions and confidence values, compute the
performance of groups of various sizes using majority and weighted majority,
and plot the average group accuracy with both methods.

Author: Davide Valeriani

"""

from cBCI.groups import performance
from cBCI.plotting import accuracy
import pandas as pd
import numpy as np

df = pd.read_csv("data/decisions.csv", index_col=0, dtype=int)

decisions = df[df.columns[df.columns.to_series().str.contains('d')]].to_numpy().T.astype(float)
decisions[decisions == -1] = np.nan
decisions -= 1
confidence = df[df.columns[df.columns.to_series().str.contains('c')]].to_numpy().T.astype(float)
confidence[confidence < 0] = np.nan
correctness = df["correct"].values-1
rts = df[df.columns[df.columns.to_series().str.contains('RT')]].to_numpy().T.astype(float)
rts = 1/rts
rts[rts == 0] = np.nan
ax = accuracy.group_accuracy(performance.majority(decisions.copy(), correctness.copy(), tie_policy="random_balanced"),
                             color="xkcd:purple", marker="o", show_sem=True, label="Majority")
ax = accuracy.group_accuracy(performance.majority(decisions.copy(), correctness.copy(), confidence.copy(), tie_policy="random_balanced"),
                             ax=ax, color="xkcd:yellowish orange", marker="s", show_sem=True, label="Confidence Majority")
ax = accuracy.group_accuracy(performance.majority(decisions.copy(), correctness.copy(), rts.copy(), tie_policy="random_balanced"),
                             ax=ax, color="xkcd:dark blue", show=True, marker="^", show_sem=True, label="RT Majority")
