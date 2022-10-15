import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import seaborn as sns

def group_accuracy(group_accuracies, ax=None, show=False, show_sem=False, print_performance=False, color="xkcd:lightblue", marker="o", ls="-", label=None):
    """Plot average group performance for groups of increasing size.

    Parameters
    ----------
    group_accuracies : list of lists
        List of accuracies of each group of a given size.
    ax : matplotlib.axes.Axes
        Axes where to plot.
    show : bool
        Shows the plotted figure at the end.
    show_sem : bool
        Plots the standard error of the mean as a shaded area.
    print_performance : bool
        Prints the average performance of each group size on standard output.
    color : str
        Color of the line. Any Matplotlib-supported color.
    marker : str
        Matplotlib marker to use.
    ls : str
        Matplotlib line style to use.
    label : str
        Label to associate to the plot.

    Returns
    -------
    matplotlib.axes.Axes
        Axes where the plot has been drawn.

    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    N = len(group_accuracies)
    x = range(1, N+1)
    means = np.zeros((N))
    sem = np.zeros((N))
    for n in range(N):
        means[n] = np.mean(group_accuracies[n])
        if len(group_accuracies[n]) > 1 and show_sem:
            sem[n] = ss.sem(group_accuracies[n])
        if print_performance:
            print("%d %d %.4f %.4f, %.4f, %4f" % (n+1, len(group_accuracies[n]), np.mean(group_accuracies[n])))
    ax.plot(x, 100*means, marker=marker, ls=ls, color=color, lw=3, ms=10, label=label)
    if show_sem:
        ax.fill_between(x, 100*(means-sem), 100*(means+sem), color=color, alpha=.2)
    ax.set_xticks(range(1, N+1))
    ax.set_ylabel("Objective accuracy (%)")
    ax.set_xlabel("Group size")
    sns.despine()
    if show:
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()
    return ax
