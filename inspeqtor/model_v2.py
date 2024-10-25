import jax
import jax.numpy as jnp
import flax.linen as nn
import optax  # type: ignore
from torch.utils.data import DataLoader
import typing

from .model import (
    VariableDict,
    calculate_metrics,
    inner_loss_v2,
    inner_loss,
    get_predict_expectation_value,
    direct_AFG_estimation,
    direct_AFG_estimation_coefficients,
    calculate_Pauli_AGF
)
from .constant import default_expectation_values_order
from .typing import HistoryEntryV2
from .physics import SimulatorFnV2, SignalParameters, avg_gate_fidelity_from_superop, to_superop


def MSEE_loss_fn(
    params: VariableDict,
    pulse_parameters: jnp.ndarray,
    unitaries: jnp.ndarray,
    expectation_values: jnp.ndarray,
    model: nn.Module,
    model_kwargs: dict = {},
):
    # Predict Vo for each pauli operator from pulse parameters
    Wos_params = model.apply(params, pulse_parameters, **model_kwargs)

    # Calculate the metrics
    metrics = calculate_metrics(
        model=model,
        model_params=params,
        pulse_parameters=pulse_parameters,
        unitaries=unitaries,
        expectation_values=expectation_values,
    )

    return (
        inner_loss(
            Wos_params=Wos_params,
            unitaries=unitaries,
            expectation_values=expectation_values,
            evaluate_expectation_values=default_expectation_values_order,
        ),
        metrics,
    )


def MAEF_loss_fn(
    params: VariableDict,
    pulse_parameters: jnp.ndarray,
    unitaries: jnp.ndarray,
    expectation_values: jnp.ndarray,
    model: nn.Module,
    model_kwargs: dict = {},
):
    # Predict Vo for each pauli operator from paluse parameters
    Wos_params = model.apply(params, pulse_parameters, **model_kwargs)
    Wos_params = ensure_wo_params_type(Wos_params)

    # Calculate the metrics
    metrics = calculate_metrics(
        model=model,
        model_params=params,
        pulse_parameters=pulse_parameters,
        unitaries=unitaries,
        expectation_values=expectation_values,
    )

    return (
        inner_loss_v2(
            Wos_params=Wos_params,
            unitaries=unitaries,
            expectation_values=expectation_values,
            evaluate_expectation_values=default_expectation_values_order,
        ),
        metrics,
    )


def create_step(
    model: nn.Module,
    optimizer: optax.GradientTransformation,
    loss_fn: typing.Callable[..., typing.Tuple[jnp.ndarray, dict]],
):

    @jax.jit
    def train_step(
        model_params: VariableDict,
        opt_state: optax.OptState,
        pulse_parameters: jnp.ndarray,
        unitaries: jnp.ndarray,
        expectation_values: jnp.ndarray,
        model_kwargs: dict = {},
    ):

        # Calculate the gradients and loss with auxiliary matrices
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            model_params,
            pulse_parameters,
            unitaries,
            expectation_values,
            model,
            model_kwargs,
        )

        # Update the optimizer state
        updates, opt_state = optimizer.update(grads, opt_state, model_params)
        # Update the model parameters
        model_params = optax.apply_updates(model_params, updates)

        return model_params, opt_state, loss, metrics

    @jax.jit
    def eval_step(
        model_params: VariableDict,
        pulse_parameters: jnp.ndarray,
        unitaries: jnp.ndarray,
        expectation_values: jnp.ndarray,
        model_kwargs: dict = {},
    ) -> typing.Tuple[jnp.ndarray, dict]:

        return loss_fn(
            model_params,
            pulse_parameters,
            unitaries,
            expectation_values,
            model,
            model_kwargs,
        )

    return train_step, eval_step


def train_model(
    train_dataloader: DataLoader,
    validation_data_loader: DataLoader,
    test_dataloader: typing.Union[DataLoader, list],
    model: nn.Module,
    optimizer: optax.GradientTransformation,
    loss_fn: typing.Callable[..., typing.Tuple[jnp.ndarray, dict]],
    model_init_key: jnp.ndarray,
    NUM_EPOCHS: int = 1500,
    callbacks: list[typing.Callable] = [],
) -> typing.Tuple[VariableDict, optax.OptState, list[HistoryEntryV2]]:

    history = []
    # Initialize the model
    sample_input = next(iter(train_dataloader))["x0"].numpy()
    model_params = model.init(model_init_key, sample_input, training=True)

    opt_state = optimizer.init(model_params)

    train_step, eval_step = create_step(model, optimizer, loss_fn)

    step = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        # Training loop
        for i, batch in enumerate(train_dataloader):

            # Get the data from the batch
            pulse_parameters = batch["x0"].numpy()
            unitaries = batch["x1"].numpy()
            expectation_values = batch["y"].numpy()

            model_params, opt_state, loss, metrics = train_step(
                model_params,
                opt_state,
                pulse_parameters,
                unitaries,
                expectation_values,
                model_kwargs={"training": True},
            )

            step += 1
            # Log the metrics to the history
            history.append(
                HistoryEntryV2(
                    MSEE=float(jnp.mean(metrics["MSE_loss"])),
                    MAEF=float(jnp.mean(metrics["AGF_loss"])),
                    step=step,
                    epoch=epoch,
                    loop="train",
                )
            )

        # Validation loop
        for i, batch in enumerate(validation_data_loader):
            pulse_parameters = batch["x0"].numpy()
            unitaries = batch["x1"].numpy()
            expectation_values = batch["y"].numpy()

            loss, metrics = eval_step(
                model_params,
                pulse_parameters,
                unitaries,
                expectation_values,
                model_kwargs={"training": False},
            )

            # Log the metrics to the history
            history.append(
                HistoryEntryV2(
                    MSEE=float(jnp.mean(metrics["MSE_loss"])),
                    MAEF=float(jnp.mean(metrics["AGF_loss"])),
                    step=step,
                    epoch=epoch,
                    loop="validation",
                )
            )

        # Test loop
        for i, batch in enumerate(test_dataloader):
            pulse_parameters = batch["x0"].numpy()
            unitaries = batch["x1"].numpy()
            expectation_values = batch["y"].numpy()

            loss, metrics = eval_step(
                model_params,
                pulse_parameters,
                unitaries,
                expectation_values,
                model_kwargs={"training": False},
            )

            # Log the metrics to the history
            history.append(
                HistoryEntryV2(
                    MSEE=float(jnp.mean(metrics["MSE_loss"])),
                    MAEF=float(jnp.mean(metrics["AGF_loss"])),
                    step=step,
                    epoch=epoch,
                    loop="test",
                )
            )

        for callback in callbacks:
            callback(
                model_params=model_params,
                opt_state=opt_state,
                history=history,
            )

    return model_params, opt_state, history


def ensure_wo_params_type(Wos_params: typing.Any) -> dict[str, dict[str, jnp.ndarray]]:
    if not isinstance(Wos_params, dict):
        raise TypeError(
            f"Expected Wos_params to be a dictionary, got {type(Wos_params)}"
        )

    for key, value in Wos_params.items():
        if not isinstance(value, dict):
            raise TypeError(
                f"Expected the values of Wos_params to be dictionaries, got {type(value)}"
            )

        for k, v in value.items():
            if not isinstance(v, jnp.ndarray):
                raise TypeError(
                    f"Expected the values of the values of Wos_params to be jnp.ndarray, got {type(v)}"
                )

    return Wos_params


def gate_loss(
    x: jnp.ndarray,
    model: nn.Module,
    model_params: VariableDict,
    simulator: SimulatorFnV2,
    array_to_list_of_params,
    target_unitary: jnp.ndarray,
):

    # x is the pulse parameters in the flattened form
    pulse_params = array_to_list_of_params(x)
    # Get the signal parameters
    signal_params = SignalParameters(
        pulse_params=pulse_params,
        phase=0,
    )
    # Calculate the unitaries
    unitaries = simulator(signal_params)

    # Evaluate model to get the Wo parameters
    Wo_params = model.apply(model_params, jnp.expand_dims(x, 0), training=False)

    # Calculate the predicted expectation values using model
    predicted_expvals = get_predict_expectation_value(
        Wo_params,
        unitaries[-2:-1],
        default_expectation_values_order,
    )

    coefficients = direct_AFG_estimation_coefficients(target_unitary)
    AGF = direct_AFG_estimation(coefficients, predicted_expvals.squeeze())
    loss = 1 - AGF

    # NOTE: For logging purposes
    fidelities_dict = calculate_Pauli_AGF(
        jax.tree.map(lambda x: jnp.squeeze(x, 0), Wo_params)
    )

    gate_fi = avg_gate_fidelity_from_superop(
        to_superop(unitaries[-1]), to_superop(target_unitary)
    )
    fidelities_dict["gate"] = gate_fi
    fidelities_dict["AGF"] = AGF

    return loss, fidelities_dict
