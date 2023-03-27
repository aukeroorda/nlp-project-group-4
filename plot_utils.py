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
    plt.plot(np.arange(1, length_losses+1), list_losses, label="loss")
    plt.title(title, fontsize=20)
    plt.xlabel("epoch", fontsize=20)
    plt.ylabel("loss", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.grid()
    return fig