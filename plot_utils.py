import numpy as np
import matplotlib.pyplot as plt

def plot_losses(list_losses, title):
    """
    ---------
    Arguments
    ---------
    list_losses: list or ndarray
        a list or a numpy array of losses
    title: str
        title for the plot

    -------
    Returns
    -------
    fig: matplotlib figure object
        returns a matplotlib figure object
    """
    length_losses = len(list_losses)
    fig = plt.figure(figsize=(12, 8))
    plt.plot(np.arange(1, length_losses+1), list_losses, "o-", label="loss")
    plt.title(title, fontsize=20)
    plt.xlabel("epoch", fontsize=20)
    plt.ylabel("loss", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.grid()
    return fig

def plot_losses_multi(list_losses, title, list_of_labels, start_epoch=0):
    """
    ---------
    Arguments
    ---------
    list_losses_1: list of lists or ndarrays
        a list of lists or numpy arrays of losses
    title: str
        title for the plot
    list_of_labels: list of strings
        a list of label strings

    -------
    Returns
    -------
    fig: matplotlib figure object
        returns a matplotlib figure object
    """
    num_plots = len(list_losses)
    length_losses = len(list_losses[0])

    fig = plt.figure(figsize=(12, 8))
    for i in range(num_plots):
        plt.plot(np.arange(start_epoch+1, length_losses+1), list_losses[i][start_epoch:], "o-", label=list_of_labels[i])
    plt.title(title, fontsize=20)
    plt.xlabel("epoch", fontsize=20)
    plt.ylabel("loss", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.grid()
    return fig
