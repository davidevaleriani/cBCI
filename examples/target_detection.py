"""
=============================================
Collaborative BCI for target detection
=============================================

In this tutorial, we show how to use cBCI package to compare group decision-making performance using a) standard
majority rule, b) a weighted majority rule, where weights are given by the decision confidence, and c) a weighted
majority rule, where weights are the reciprocal of the reaction time.
The script uses cBCI.groups module to compute average group performance across different group sizes, and the
cBCI.plotting module to visualize the accuracy as a line plot.

Author: Davide Valeriani

"""

from cBCI.groups import performance
from cBCI.plotting import accuracy
import pandas as pd
import numpy as np

# Load dataset using pandas
df = pd.read_csv("data/decisions.csv", index_col=0, dtype=int)

# Data parsing
# Extract decisions, confidence and correctness from pandas DataFrame
decisions = df[df.columns[df.columns.to_series().str.contains('d')]].to_numpy().T.astype(float)
decisions[decisions == -1] = np.nan
# Normalize decisions so that 0 means incorrect, 1 means correct
decisions -= 1
confidence = df[df.columns[df.columns.to_series().str.contains('c')]].to_numpy().T.astype(float)
confidence[confidence < 0] = np.nan
# Normalize correctness so that 0 means incorrect, 1 means correct
correctness = df["correct"].values-1
# Extract reaction times, as additional ways to measure confidence
rts = df[df.columns[df.columns.to_series().str.contains('RT')]].to_numpy().T.astype(float)
# Replace RT=0 with nan
rts[rts == 0] = np.nan
# Compute 1/RT, so that high RT mean low confidence
rts = 1/rts
# Calculate performance of majority and plot group accuracy
# Note how we can set typical plotting parameters as in matplotlib
ax = accuracy.group_accuracy(performance.majority(decisions.copy(), correctness.copy(), tie_policy="random_balanced"),
                             color="xkcd:purple", marker="o", show_sem=True, label="Majority")
# Calculate performance of confidence majority and plot group accuracy
ax = accuracy.group_accuracy(performance.majority(decisions.copy(), correctness.copy(), confidence.copy(),
                                                  tie_policy="random_balanced"),
                             ax=ax, color="xkcd:yellowish orange", marker="s", show_sem=True,
                             label="Confidence Majority")
# Calculate performance of RT confidence majority and plot group accuracy
ax = accuracy.group_accuracy(performance.majority(decisions.copy(), correctness.copy(), rts.copy(),
                                                  tie_policy="random_balanced"),
                             ax=ax, color="xkcd:dark blue", show=True, marker="^", show_sem=True, label="RT Majority")
