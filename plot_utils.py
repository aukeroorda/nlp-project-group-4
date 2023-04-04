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

def plot_losses_2(list_losses_1, list_losses_2, title, label_1, label_2):
    """
    ---------
    Arguments
    ---------
    list_losses_1: list or ndarray
        a list or a numpy array of losses
    list_losses_2: list or ndarray
        a list or a numpy array of losses
    title: str
        title for the plot

    -------
    Returns
    -------
    fig: matplotlib figure object
        returns a matplotlib figure object
    """
    length_losses = len(list_losses_1)
    fig = plt.figure(figsize=(12, 8))
    plt.plot(np.arange(1, length_losses+1), list_losses_1, "o-", label=label_1)
    plt.plot(np.arange(1, length_losses+1), list_losses_2, "o-", label=label_2)
    plt.title(title, fontsize=20)
    plt.xlabel("epoch", fontsize=20)
    plt.ylabel("loss", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.grid()
    return fig
