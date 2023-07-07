# -*- coding: utf-8 -*-
"""
-> Matplotlib custom operations
"""

import matplotlib.pyplot as plt


def despine(ax: plt.Axes, grid: bool | None = False):
    """
    -> Hides top and right axes, and thickens bottom and left axes.
    """
    if grid:
        ax.spines["top"].set_color("darkgrey")
        ax.spines["right"].set_color("darkgrey")
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(1.25)
    ax.spines["bottom"].set_linewidth(1.25)
    ax.tick_params(width=1.25)


def set_pyplot_params():
    """
    -> Custom matplotlib defaults for font.
    """
    plt.rcParams.update({"font.family": "monospace"})
    # plt.rcParams.update({"font.family": "Consolas"})
    # plt.rcParams.update({"font.family": "Fira Code"})
    plt.rcParams.update({"font.size": 9})
