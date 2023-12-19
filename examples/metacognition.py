"""
=============================================
Metacognitive accuracy calculation
=============================================

In this tutorial, we see how to calculate the metacognitive accuracy of each participant in a decision-making experiment
from their decision and confidence values. Metacognition is calculated as in this paper:
https://doi.org/10.3389/fnhum.2014.00443

Author: Davide Valeriani

"""

from cBCI.individual.performance import metacognitive_accuracy
import pandas as pd
import numpy as np

# Load data with sample decisions and confidence values
df = pd.read_csv("data/decisions.csv", index_col=0, dtype=int)

# Extract correct response of each trial: this is the same for all participants
correctness = df["correct"].values
df = df.drop(["correct"], axis=1)
# Extract decisions
decisions = df[df.columns[df.columns.to_series().str.contains('d')]].to_numpy().T
# Extract confidence
confidence = df[df.columns[df.columns.to_series().str.contains('c')]].to_numpy().T
# Normalize confidence to 0-1 range, as it is expressed as confidence={1,2,3,4} in the experiment.
confidence = (confidence - 1) / 3
confidence[confidence < 0] = np.nan

# Iterate through participants
for s in range(decisions.shape[0]):
    # Calculate accuracy of each decision
    accuracy = (decisions[s] == correctness).astype(int)
    # Calculate metacognition of each trial
    metacog = metacognitive_accuracy(accuracy, confidence[s])
    print(f"\nSubject {s}\nMetacognition in each trial:\n{metacog}\nAverage metacognition = {np.nanmean(metacog):.2f}%")
