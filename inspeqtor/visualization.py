import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import typing
import pandas as pd  # type: ignore
from .constant import default_expectation_values_order


def draw_complex_pulse(
    waveform: jnp.ndarray,
    x_axis: jnp.ndarray,
    ax: typing.Union[Axes, None] = None,
    font_size: int = 12,
):

    has_ax = ax is not None

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    assert ax is not None

    ax.bar(x_axis, jnp.real(waveform), label="real", color="orange", alpha=0.5)
    ax.bar(x_axis, jnp.imag(waveform), label="imag", color="blue", alpha=0.5)

    ax.set_xlabel("Time", fontsize=font_size)

    # force integer ticks
    # ax.set_xticks(x_axis)
    # Text size
    ax.tick_params(axis="both", labelsize=font_size)

    ax.set_xlabel("Time (dt)", fontsize=font_size)
    ax.set_ylabel("Amplitude", fontsize=font_size)

    ax.legend(fontsize=font_size)

    if not has_ax:

        fig.tight_layout()

        return fig, ax
    else:
        return None, ax


def plot_history(
    history: list,
    lr_schedule: typing.Union[typing.Callable[[int], float], None] = None,
    window_size: int = 50,
    with_epoch_loss: bool = False,
    font_size: int = 12,
):

    hist_df = pd.DataFrame(history)
    train = hist_df[["global_step", "batch_loss"]].values

    train_x = train[:, 0]
    train_y = train[:, 1]

    validate = hist_df[["global_step", "val_loss"]].replace(0, jnp.nan).dropna().values

    validate_x = validate[:, 0]
    validate_y = validate[:, 1]

    # Determine the number of subplots
    num_subplots = 1 if lr_schedule is None else 2
    height_ratios = [2, 1] if lr_schedule is not None else [1]

    # The second plot has height ratio 2
    fig, axes = plt.subplots(
        num_subplots, 1, figsize=(10, 6), sharex=True, height_ratios=height_ratios
    )

    axes = [axes] if num_subplots == 1 else axes

    # The first plot is the training loss and the validation loss
    axes[0].plot(
        train_x, train_y, label="Training loss (MSE)", color="#6a82fb", alpha=0.25
    )
    axes[0].plot(
        validate_x,
        validate_y,
        label="Validation loss (MSE)",
        color="#fc5c7d",
        alpha=0.25,
    )

    if with_epoch_loss:

        epoch = (
            hist_df[["global_step", "epoch_loss"]].replace(0, jnp.nan).dropna().values
        )
        epoch_x = epoch[:, 0]
        epoch_y = epoch[:, 1]
        axes[0].plot(epoch_x, epoch_y, label="epoch_loss", color="#a6e22e", alpha=0.25)

    # Set the y-axis to log scale
    axes[0].set_yscale("log")

    # Calculate the moving average
    ma_train_y = pd.Series(train_y).rolling(window_size).mean().values
    ma_validate_y = pd.Series(validate_y).rolling(window_size).mean().values

    # plot the moving average
    axes[0].plot(
        train_x, ma_train_y, label="moving average Training loss (MSE)", color="#6a82fb"
    )
    axes[0].plot(
        validate_x,
        ma_validate_y,
        label="moving average Validation loss (MSE)",
        color="#fc5c7d",
    )

    if with_epoch_loss:
        ma_epoch_y = pd.Series(epoch_y).rolling(window_size).mean().values

        axes[0].plot(
            epoch_x, ma_epoch_y, label="moving average epoch_loss", color="#a6e22e"
        )

    # From the data calculate the scale of the y-axis that is needed to plot the horizontal line
    y_min = min(min(train_y), min(validate_y))
    y_max = max(max(train_y), max(validate_y))

    max_order = int(jnp.log10(y_max))
    min_order = int(jnp.log10(y_min))

    hlines = [10**i for i in range(min_order, max_order + 1)]

    for hline in hlines:
        axes[0].axhline(hline, color="red", linestyle="--", alpha=0.25)

    axes[0].legend(fontsize=font_size)

    if lr_schedule is not None:
        lrs = [lr_schedule(step) for step in train_x]
        axes[1].plot(train_x, lrs, label="learning_rate", color="#6a82fb")
        axes[1].legend(fontsize=font_size)
        axes[1].tick_params(axis="both", which="major", labelsize=font_size)

    # X-axis label
    axes[0].set_xlabel("Global iteration", fontsize=font_size)
    axes[0].set_ylabel("loss", fontsize=font_size)
    axes[0].tick_params(axis="both", which="major", labelsize=font_size)

    fig.tight_layout()

    return fig, axes


def plot_expvals(expvals):
    fig, axes = plt.subplots(3, 6, figsize=(20, 10), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flatten()):
        ax.plot(expvals[i])
        ax.set_title(
            f"{default_expectation_values_order[i].initial_state}/{default_expectation_values_order[i].observable}"
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
    for idx, exp in enumerate(default_expectation_values_order):

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
