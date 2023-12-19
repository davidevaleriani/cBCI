import numpy as np
from itertools import combinations
from pingouin import wilcoxon, compute_effsize
import pandas as pd
from typing import List, Optional, LiteralString
from numpy.typing import NDArray


def majority(decisions: NDArray, correctness: NDArray, weights: Optional[NDArray] = None,
             tie_policy: LiteralString = 'random', max_group_size: Optional[int] = None) -> List[List]:
    """Computes the performance of all groups of size from 1 to N that it is
    possible to assemble with the N subjects in the experiment using weighted
    majority.

    Parameters
    ----------
    decisions : 2D array with decisions of each subject in the experiment.
        It should have shape SUBJECTS x TRIALS.
        Missing responses should be denoted with np.nan and will be ignored for
        that group. If all members of the group have missed the response in one trial,
        that trial will be ignored.
    correctness : list or 1D array with the correct response of each trial.
        It should be the same for all subjects.
        If not, subjects' responses should be reordered accordingly.
    weights : 2D array with the weights to associate to each decision.
        If None, standard majority will be used (weights constant).
        It should have shape SUBJECTS x TRIALS.
        Missing weights should be denoted with np.nan and will be considered as 0.
    tie_policy : string denoting the policy to use to break ties.
        If "random", randomly choose a decision between tied responses.
        If "random_balanced", 1/M of the ties will be considered correct group
        responses, where M are the available options.
    max_group_size : int representing the maximum group size to simulate.
        If None, becomes equal to the number of subjects.

    Returns
    -------
    List of lists
        Accuracy of each simulated group for various sizes.

    """
    # Check we have correctness information for each trial for which we have decisions
    assert decisions.shape[1] == correctness.shape[0]
    # Ensure all decisions are either 0, 1, or nan (decision not reported)
    assert np.all(np.logical_or(np.logical_or(decisions == 0, decisions == 1), np.isnan(decisions)))
    if weights is not None:
        if np.any(weights < 0):
            print("WARNING: negative values in the weights")
        # Make all weights as positive numbers
        weights = np.clip(weights, 1e-9, np.inf)
        # Nan weights will be replaced with 0, so the decision will be ignored
        weights[np.isnan(weights)] = 0
    num_people = decisions.shape[0]
    num_trials = decisions.shape[1]
    if max_group_size is None:
        max_group_size = num_people
    elif max_group_size > num_people:
        # If the max group size specified is bigger than the number of people, issue a warning and reset to num_people
        print("Warning: max_group_size greater than the number of subjects. Resetting it to %d" % num_people)
        max_group_size = num_people
    elif max_group_size < 2:
        # If the max group size is 1 or less, issue a warning and reset to num_people
        print("Warning: max_group_size smaller than 2. Resetting it to %d" % num_people)
        max_group_size = num_people
    # Relabel decisions and correctness to be -1 or +1
    decisions[decisions == 0] = -1
    correctness[correctness == 0] = -1
    # Calculate the accuracy of all possible groups of different sizes
    group_accuracies = []
    for group_size in range(1, max_group_size+1):
        # Generate groups
        groups = combinations(range(num_people), group_size)
        group_accuracies.append([])
        for group in groups:
            group_decisions = []
            for trial in range(num_trials):
                votes = decisions[group, trial]
                if weights is not None and group_size > 1:
                    w = weights[group, trial]
                    w = w[~np.isnan(votes)]
                    # Discard nan decisions from votes
                    votes = w*votes[~np.isnan(votes)]
                else:
                    # Discard nan decisions from votes
                    votes = votes[~np.isnan(votes)]
                # The sum of the votes of all group members will be negative if the majority chooses option 0,
                # or positive if the majority chooses option 1. Hence, we take the sign of this number to represent
                # the group decision.
                if len(votes) > 0:
                    score = np.sign(np.sum(votes))
                    group_decisions.append(score)
                else:
                    # All votes are nan: skip
                    group_decisions.append(np.nan)
                    continue
            group_decisions = np.array(group_decisions)
            accuracy = (np.sum([1 if group_decisions[t] == correctness[t] else 0
                                for t in range(num_trials)]) / (num_trials))
            # Break ties and correct group accuracy
            ties = np.where(group_decisions == 0)[0]
            num_ties = len(ties)
            if tie_policy == "random_balanced":
                # Half of the ties will be replaced by correct decisions
                accuracy += 0.5 * (num_ties / num_trials)
            elif tie_policy == "random":
                # Random decision at every tie
                for tie in range(num_ties):
                    accuracy += (1 / num_trials) if np.random.random() < 0.5 else 0
            group_accuracies[-1].append(accuracy)
    return group_accuracies


def statistics(group_accuracies_x: List[List], group_accuracies_y: List[List], test: str = "wilcoxon") -> pd.DataFrame:
    """Calculate statistics between group accuracies using pingouin library.

    Parameters
    ----------
    group_accuracies_x : list of lists
        Accuracy of simulated groups of different sizes using a certain method.
    group_accuracies_y : list of lists
        Accuracy of simulated groups of different sizes using a different method.
    test : str
        Statistical test to use to compare the two accuracies.
        Wilcoxon signed-rank test is currently supported.

    Returns
    -------
    pd.DataFrame
        Statistics for different group sizes.

    """
    stats = None
    for group_size in range(len(group_accuracies_x)):
        # Ensure we have the same number of groups for the two methods
        assert len(group_accuracies_x[group_size]) == len(group_accuracies_y[group_size])
        if test == "wilcoxon":
            # Calculate wilcoxon signed-rank test
            res = wilcoxon(group_accuracies_x[group_size], group_accuracies_y[group_size], **{'zero_method': 'zsplit'})
            # Calculate effect size
            res["effect_size"] = compute_effsize(group_accuracies_x[group_size], group_accuracies_y[group_size])
            # Add results to stats dataframe
            stats = res if stats is None else pd.concat([stats, res], ignore_index=True)
        else:
            raise ValueError("Test %s not implemented" % test)
    return stats


def auroc2(accuracies: List, confidences: List, number_of_confidence_ratings: int) -> float:
    """Compute area under type 2 ROC to assess calibration of confidence ratings for a given experiment and subject.

    Parameters
    ----------
    accuracies : list
        List of decisions, with 1 indicating correct decisions, 0 indicating errors.
    confidences : list
        Confidence ratings associated to each decision.
    number_of_confidence_ratings : int
        Number of confidence levels that are available. Assuming they go from 1 to this number.
        So, if levels are 1,2,3,4, then pass number_of_confidence_ratings=4.

    Returns
    -------
    float
       Area under the type 2 ROC.
    """
    assert len(confidences) == len(accuracies)
    # Cast inputs to numpy array
    accuracies = np.array(accuracies)
    confidences = np.array(confidences)
    # Calculate hit rate and false alarm rate
    hit_rates = np.zeros(number_of_confidence_ratings)
    false_alarm_rates = np.zeros(number_of_confidence_ratings)
    for c in range(1, number_of_confidence_ratings+1):
        hit_rates[number_of_confidence_ratings-c] = np.sum([(accuracies == 1) & (confidences == c)])+0.5
        false_alarm_rates[number_of_confidence_ratings-c] = np.sum([(accuracies == 0) & (confidences == c)])+0.5
    # Normalize hit and false alarm rates
    hit_rates /= np.sum(hit_rates)
    false_alarm_rates /= np.sum(false_alarm_rates)
    # Calculate cumulative sum
    cumsum_hit = [0]+list(np.cumsum(hit_rates))
    cumsum_false_alarm = [0]+list(np.cumsum(false_alarm_rates))
    k = []
    for c in range(number_of_confidence_ratings):
        k.append((cumsum_hit[c+1]-cumsum_false_alarm[c])**2-(cumsum_hit[c]-cumsum_false_alarm[c+1])**2)
    auroc2 = 0.5 + 0.25*np.sum(k)
    return auroc2


if __name__ == "__main__":
    decisions = np.array([[1, 1],
                          [1, 1],
                          [np.nan, 0]])
    correctness = np.array([1, 1])
    accuracy = majority(decisions, correctness, tie_policy='random_balanced')
    print(accuracy)