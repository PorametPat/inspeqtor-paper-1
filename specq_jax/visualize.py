from typing import Callable, Union
import matplotlib.pyplot as plt
import pandas as pd
import jax.numpy as jnp
from specq_dev import shared # type: ignore

def plot_history(
    history: list, lr_schedule: Union[Callable[[int], float], None] = None
):

    hist_df = pd.DataFrame(history)
    train = hist_df[["global_step", "loss"]].values

    train_x = train[:, 0]
    train_y = train[:, 1]

    validate = hist_df[["global_step", "val_loss"]].replace(0, jnp.nan).dropna().values

    validate_x = validate[:, 0]
    validate_y = validate[:, 1]

    # Determine the number of subplots
    num_subplots = 1 if lr_schedule is None else 2
    height_ratios = [2, 1] if lr_schedule is not None else [1]

    # The second plot has height ratio 2
    fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 6), sharex=True, height_ratios=height_ratios)

    axes = [axes] if num_subplots == 1 else axes

    # The first plot is the training loss and the validation loss
    axes[0].plot(train_x, train_y, label="train_loss", color="#6a82fb")
    axes[0].plot(validate_x, validate_y, label="val_loss", color="#fc5c7d")
    axes[0].set_yscale("log")

    # plot the horizontal line [1e-4, 1e-3, 1e-2]
    axes[0].axhline(1e-4, color="red", linestyle="--")
    axes[0].axhline(1e-3, color="red", linestyle="--")
    axes[0].axhline(1e-2, color="red", linestyle="--")

    axes[0].legend()

    if lr_schedule is not None:
        lrs = [lr_schedule(step) for step in train_x]
        axes[1].plot(train_x, lrs, label="learning_rate", color="#6a82fb")
        axes[1].legend()

    fig.tight_layout()

    return fig, axes

def plot_expvals(expvals):
    fig, axes = plt.subplots(3, 6, figsize=(20, 10), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flatten()):
        ax.plot(expvals[i])
        ax.set_title(
            f"{shared.default_expectation_values[i].initial_state}/{shared.default_expectation_values[i].observable}"
        )

    # Set the ylim to (-1.05, 1.05)
    for ax in axes.flatten():
        ax.set_ylim(-1.05, 1.05)

    # Set title for the figure
    fig.suptitle("Expectation values for the unitaries")

    plt.tight_layout()
    plt.show()


def plot_expvals_v2(expvals):
    fig, axes = plt.subplot_mosaic(
        """
        +r0
        -l1
        """,
        figsize=(10, 5),
        sharex=True,
        sharey=True,
    )

    expvals_dict = {}
    for idx, exp in enumerate(shared.default_expectation_values):

        if exp.initial_state not in expvals_dict:
            expvals_dict[exp.initial_state] = {}

        expvals_dict[exp.initial_state][exp.observable] = expvals[idx]

    for idx, (initial_state, expvals) in enumerate(expvals_dict.items()):
        ax = axes[initial_state]
        for observable, expval in expvals.items():
            ax.plot(expval, label=observable)
        ax.set_title(f"Initial state: {initial_state}")
        ax.set_ylim(-1.05, 1.05)
        ax.legend()

    # Set title for the figure
    fig.suptitle("Expectation values for the unitaries")

    plt.tight_layout()

    return fig