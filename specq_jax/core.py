import jax
import jax.numpy as jnp
import specq_dev.shared as specq  # type: ignore
import pennylane as qml  # type: ignore
from typing import Callable, Union, Protocol, Any
from torch.utils.data import Dataset, DataLoader
from jaxtyping import Array, Complex, Float
from flax import linen as nn
import optax  # type: ignore
import matplotlib.pyplot as plt
from specq_dev.jax import JaxBasedPulseSequence  # type: ignore
import pandas as pd
from alive_progress import alive_bar  # type: ignore
from jaxopt import ProjectedGradient  # type: ignore
from jaxopt.projection import projection_box  # type: ignore
from dataclasses import dataclass
from flax.typing import VariableDict
from .simulator import SimulatorFnV2, SignalParameters
from specq_dev.shared import ParametersDictType


class LossFn(Protocol):

    def __call__(
        self,
        params: VariableDict,
        pulse_parameters: Float[Array, "batch num_pulses num_features"],  # noqa: F722
        unitaries: Complex[Array, "batch dim dim"],  # noqa: F722
        expectation_values: Float[Array, "batch num_expectations"],  # noqa: F722
        training: bool,
    ) -> Float[Array, "1"]: ...


class TrainStepFn(Protocol):

    def __call__(
        self,
        params: VariableDict,
        opt_state: optax.OptState,
        pulse_parameters: Float[Array, "batch num_pulses num_features"],  # noqa: F722
        unitaries: Complex[Array, "batch dim dim"],  # noqa: F722
        expectations: Float[Array, "batch num_expectations"],  # noqa: F722
        dropout_key: jnp.ndarray,
        transform_state: Union[None, optax.OptState],
    ) -> tuple[Any, Any, float]: ...


class TestStepFn(Protocol):

    def __call__(
        self,
        params: VariableDict,
        pulse_parameters: Float[Array, "batch num_pulses num_features"],  # noqa: F722
        unitaries: Complex[Array, "batch dim dim"],  # noqa: F722
        expectations: Float[Array, "batch num_expectations"],  # noqa: F722
    ) -> float: ...


@dataclass
class HistoryEntry:
    epoch: int
    step: int
    loss: float
    global_step: int
    val_loss: float | None = None
    lr: float | None = None


class CallbackFn(Protocol):
    def __call__(
        self,
        params: VariableDict,
        opt_state: optax.OptState,
        history: list[HistoryEntry],
    ) -> None: ...


def calculate_exp(
    unitary: jnp.ndarray, operator: jnp.ndarray, initial_state: jnp.ndarray
) -> jnp.ndarray:
    density_matrix = jnp.outer(initial_state, initial_state.T.conj())
    rho = jnp.matmul(unitary, jnp.matmul(density_matrix, unitary.T.conj()))
    temp = jnp.matmul(rho, operator)
    return jnp.real(jnp.trace(temp))


def mse(y_true, y_pred):
    return jnp.mean(jnp.square(y_true - y_pred))


batched_calculate_expectation_value = jax.vmap(calculate_exp, in_axes=(0, 0, None))
batch_mse = jax.vmap(mse, in_axes=(0, 0))


def Wo_2_level(
    U: jnp.ndarray,
    D: jnp.ndarray,
) -> jnp.ndarray:

    alpha = U[0]
    theta = U[1]
    beta = U[2]

    lambda_1 = D[0]
    lambda_2 = D[1]

    q_00 = jnp.exp(1j * alpha) * jnp.cos(theta)
    q_01 = jnp.exp(1j * beta) * jnp.sin(theta)
    q_10 = jnp.exp(-1j * beta) * jnp.sin(theta)
    q_11 = -jnp.exp(-1j * alpha) * jnp.cos(theta)

    Q = jnp.array([[q_00, q_01], [q_10, q_11]])

    _D = jnp.array([[lambda_1, 0], [0, lambda_2]])

    return Q @ _D @ Q.T.conj()


def get_predict_expectation_value(
    Wos_params: dict[str, dict[str, jnp.ndarray]],
    unitaries: jnp.ndarray,
    evaluate_expectation_values: list[specq.ExpectationValue],
) -> jnp.ndarray:

    predict_expectation_values = []

    # Calculate expectation values for all cases
    for _, exp_case in enumerate(evaluate_expectation_values):
        Wo = jax.vmap(Wo_2_level, in_axes=(0, 0))(
            Wos_params[exp_case.observable]["U"], Wos_params[exp_case.observable]["D"]
        )
        # Calculate expectation value for each pauli operator
        batch_expectaion_values = batched_calculate_expectation_value(
            unitaries,
            Wo,
            jnp.array(exp_case.initial_statevector),
        )

        predict_expectation_values.append(batch_expectaion_values)

    return jnp.array(predict_expectation_values)


def inner_loss(Wos_params, unitaries, expectation_values, evaluate_expectation_values):

    predict_expectation_values = get_predict_expectation_value(
        Wos_params=Wos_params,
        unitaries=unitaries,
        evaluate_expectation_values=evaluate_expectation_values,
    )

    return jnp.mean(batch_mse(expectation_values, predict_expectation_values.T))


def loss_fn_with_dropout(
    params: VariableDict,
    pulse_parameters: Float[Array, "batch num_pulses num_features"],  # noqa: F722
    unitaries: Complex[Array, "batch dim dim"],  # noqa: F722
    expectation_values: Complex[Array, "batch num_expectations"],  # noqa: F722
    training: bool,
    dropout_key: jnp.ndarray,
    model: nn.Module,
    evaluate_expectation_values: list[
        specq.ExpectationValue
    ] = specq.default_expectation_values,
):
    # Predict Vo for each pauli operator from paluse parameters
    Wos_params = model.apply(
        params, pulse_parameters, training=training, rngs={"dropout": dropout_key}
    )

    return inner_loss(
        Wos_params=Wos_params,
        unitaries=unitaries,
        expectation_values=expectation_values,
        evaluate_expectation_values=evaluate_expectation_values,
    )


def loss_fn(
    params: VariableDict,
    pulse_parameters: Float[Array, "batch num_pulses num_features"],  # noqa: F722
    unitaries: Complex[Array, "batch dim dim"],  # noqa: F722
    expectation_values: Complex[Array, "batch num_expectations"],  # noqa: F722
    training: bool,
    model: nn.Module,
    evaluate_expectation_values: list[
        specq.ExpectationValue
    ] = specq.default_expectation_values,
):
    # Predict Vo for each pauli operator from paluse parameters
    Wos_params = model.apply(params, pulse_parameters, training=training)

    return inner_loss(
        Wos_params=Wos_params,
        unitaries=unitaries,
        expectation_values=expectation_values,
        evaluate_expectation_values=evaluate_expectation_values,
    )


def evaluate_accuracy(
    model_params: VariableDict,
    pulse_parameters: Array,
    unitaries: Array,
    expectations: Array,
    indices: Array,
    tobe_trained_model: nn.Module,
):

    loss = loss_fn(
        params=model_params,
        pulse_parameters=pulse_parameters[indices],
        unitaries=unitaries[indices],
        expectation_values=expectations[indices],
        training=False,
        model=tobe_trained_model,
    )
    return loss


def create_train_step(
    key: jnp.ndarray,
    model: nn.Module,
    optimiser: optax.GradientTransformationExtraArgs,
    input_shape: tuple[int, int],
    lr_transformer: Union[None, optax.GradientTransformationExtraArgs] = None,
):
    params = model.init(
        key,
        jnp.ones(input_shape, jnp.float32),
        training=False,
    )  # dummy key just as example input
    opt_state = optimiser.init(params)

    if lr_transformer is not None:
        lr_transformer_state = lr_transformer.init(opt_state)
    else:
        lr_transformer_state = None

    @jax.jit
    def train_step(
        params: VariableDict,
        opt_state: optax.OptState,
        pulse_parameters: Float[Array, "batch num_pulses num_features"],  # noqa: F722
        unitaries: Complex[Array, "batch dim dim"],  # noqa: F722
        expectations: Complex[Array, "batch num_expectations"],  # noqa: F722
        dropout_key: jnp.ndarray,
        lr_transformer_state: None,
    ):
        loss, grads = jax.value_and_grad(loss_fn_with_dropout)(
            params,
            pulse_parameters,
            unitaries,
            expectations,
            True,
            dropout_key,
            model=model,
        )

        updates, opt_state = optimiser.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss

    @jax.jit
    def train_step_wtih_lr_transformer(
        params: VariableDict,
        opt_state: optax.OptState,
        pulse_parameters: Float[Array, "batch num_pulses num_features"],  # noqa: F722
        unitaries: Complex[Array, "batch dim dim"],  # noqa: F722
        expectations: Complex[Array, "batch num_expectations"],  # noqa: F722
        dropout_key: jnp.ndarray,
        lr_transformer_state: optax.OptState,
    ):
        loss, grads = jax.value_and_grad(loss_fn_with_dropout)(
            params,
            pulse_parameters,
            unitaries,
            expectations,
            True,
            dropout_key,
            model=model,
        )

        updates, opt_state = optimiser.update(grads, opt_state, params)

        # Apply the lr_transformer
        updates = optax.tree_utils.tree_scalar_mul(lr_transformer_state.scale, updates)

        params = optax.apply_updates(params, updates)

        return params, opt_state, loss

    @jax.jit
    def test_step(
        params: VariableDict,
        pulse_parameters: Float[Array, "batch num_pulses num_features"],  # noqa: F722
        unitaries: Complex[Array, "batch dim dim"],  # noqa: F722
        expectations: Complex[Array, "batch num_expectations"],  # noqa: F722
    ):
        loss = loss_fn(
            params,
            pulse_parameters,
            unitaries,
            expectations,
            training=False,
            model=model,
        )
        return loss

    return (
        train_step_wtih_lr_transformer if lr_transformer is not None else train_step,
        test_step,
        params,
        opt_state,
        lr_transformer_state,
    )


def with_validation_train(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    train_step: TrainStepFn,
    test_step: TestStepFn,
    model_params: VariableDict,
    opt_state: optax.OptState,
    dropout_key: jnp.ndarray,
    num_epochs=1250,
    force_tty=True,
    lr_transformer: Union[None, optax.GradientTransformationExtraArgs] = None,
    lr_transformer_state: Union[None, optax.OptState] = None,
    callback: Union[CallbackFn, None] = None,
):

    history = []
    total_len = len(train_dataloader)

    NUM_EPOCHS = num_epochs

    lr_transformer_update = (
        jax.jit(lr_transformer.update) if lr_transformer is not None else None
    )

    with alive_bar(int(NUM_EPOCHS * total_len), force_tty=force_tty) as bar:
        for epoch in range(NUM_EPOCHS):
            total_loss = 0.0
            for i, batch in enumerate(train_dataloader):

                key, dropout_key = jax.random.split(dropout_key)

                _pulse_parameters = batch["x0"].numpy()
                _unitaries = batch["x1"].numpy()
                _expectations = batch["y"].numpy()

                model_params, opt_state, loss = train_step(
                    model_params,
                    opt_state,
                    _pulse_parameters,
                    _unitaries,
                    _expectations,
                    key,
                    lr_transformer_state,
                )

                history.append(
                    HistoryEntry(
                        epoch=epoch,
                        step=i,
                        loss=float(loss),
                        global_step=epoch * total_len + i,
                    )
                )

                total_loss += loss

                bar()

            # Validation
            val_loss = 0.0
            for i, batch in enumerate(val_dataloader):

                _pulse_parameters = batch["x0"].numpy()
                _unitaries = batch["x1"].numpy()
                _expectations = batch["y"].numpy()

                val_loss += test_step(
                    model_params, _pulse_parameters, _unitaries, _expectations
                )

            history[-1].val_loss = float(val_loss / len(val_dataloader))

            if lr_transformer is not None and lr_transformer_update is not None:
                # Adjust the learning rate
                _, transform_state = lr_transformer_update(
                    updates=model_params,
                    state=lr_transformer_state,
                    value=history[-1].val_loss,
                )

                history[-1].lr = float(transform_state.scale)

            if callback is not None:
                callback(model_params, opt_state, history)

    return model_params, opt_state, history


def default_adaptive_lr_transform(
    PATIENCE: int,
    COOLDOWN: int,
    FACTOR: float,
    RTOL: float,
    ACCUMULATION_SIZE: int,
):
    """The default adaptive learning rate transform.

    Args:
        PATIENCE (int): Number of epochs with no improvement after which learning rate will be reduced.
        COOLDOWN (int): Number of epochs to wait before resuming normal operation after the learning rate reduction.
        FACTOR (float): Factor by which to reduce the learning rate.
        RTOL (float): Relative tolerance for measuring the new optimum.
        ACCUMULATION_SIZE (int): Number of iterations to accumulate an average value.
    """
    transform = optax.contrib.reduce_on_plateau(
        patience=PATIENCE,
        cooldown=COOLDOWN,
        factor=FACTOR,
        rtol=RTOL,
        accumulation_size=ACCUMULATION_SIZE,
    )
    return transform


def linear_schedule_with_warmup(
    steps: int = 10000,
    warmup_steps: int = 1000,
    warmup_start_lr: float = 1e-6,
    start_lr: float = 1e-2,
    end_lr: float = 1e-6,
):
    lr_scheduler = optax.join_schedules(
        [
            optax.linear_schedule(
                warmup_start_lr,
                start_lr,
                warmup_steps,
            ),
            optax.linear_schedule(
                start_lr,
                end_lr,
                steps - warmup_steps,
            ),
        ],
        [warmup_steps],
    )
    return lr_scheduler


X, Y, Z = (
    jnp.array(specq.Operator.from_label("X")),
    jnp.array(specq.Operator.from_label("Y")),
    jnp.array(specq.Operator.from_label("Z")),
)


def gate_loss(
    x: jnp.ndarray,
    model: nn.Module,
    model_params: VariableDict,
    simulator: SimulatorFnV2,
    pulse_sequence: JaxBasedPulseSequence,
    target_unitary: jnp.ndarray,
):
    fidelities_dict: dict[str, jnp.ndarray] = {}
    # x is the pulse parameters in the flattened form
    # pulse_params = array_to_params_list(x)
    pulse_params = pulse_sequence.array_to_list_of_params(x)

    # Get the waveforms
    # waveforms = pulse_sequence.get_waveform(pulse_params)
    signal_params = SignalParameters(
        pulse_params=pulse_params,
        phase=0,
    )

    # Get the unitaries
    unitaries = simulator(signal_params)

    # Evaluate model to get the Wo parameters
    Wo_params = model.apply(model_params, jnp.expand_dims(x, 0), training=False)
    # squeeze the Wo_params
    Wo_params = jax.tree.map(lambda x: jnp.squeeze(x, 0), Wo_params)

    loss = jnp.array(0.0)
    # Get each Wo for each Pauli operator
    for pauli_str, pauli_op in zip(["X", "Y", "Z"], [X, Y, Z]):
        Wo = Wo_2_level(U=Wo_params[pauli_str]["U"], D=Wo_params[pauli_str]["D"])
        # evaluate the fidleity to the Pauli operator
        value = gate_fidelity(pauli_op, Wo)
        fidelities_dict[pauli_str] = value

        loss += (1 - value) ** 2

    gate_fi = gate_fidelity(target_unitary, unitaries[-1])
    fidelities_dict["gate"] = gate_fi

    # Calculate the loss
    loss += (1 - gate_fi) ** 2

    return loss


def optimize(
    x0: jnp.ndarray,
    lower: list[ParametersDictType],
    upper: list[ParametersDictType],
    fun: Callable[[jnp.ndarray], jnp.ndarray],
):

    pg = ProjectedGradient(fun=fun, projection=projection_box)
    opt_params, state = pg.run(jnp.array(x0), hyperparams_proj=(lower, upper))

    return opt_params, state


def evalulate_model_to_pauli(model: nn.Module, model_params, pulse_params):

    Wo_params = model.apply(model_params, pulse_params, training=False)

    assert isinstance(Wo_params, dict)

    fidelities = {}
    for pauli_str, pauli_op in zip(["X", "Y", "Z"], [X, Y, Z]):

        if isinstance(Wo_params[pauli_str], dict):
            Wos = jax.vmap(Wo_2_level, in_axes=(0, 0))(
                Wo_params[pauli_str]["U"], Wo_params[pauli_str]["D"]
            )
        else:
            assert isinstance(Wo_params[pauli_str], jnp.ndarray)
            Wos = Wo_params[pauli_str]
        _fidel = jax.vmap(gate_fidelity, in_axes=(0, None))(Wos, pauli_op)
        fidelities[pauli_str] = _fidel

    return fidelities


def gate_fidelity(U: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:

    up = jnp.trace(U.conj().T @ V)
    down = jnp.sqrt(jnp.trace(U.conj().T @ U) * jnp.trace(V.conj().T @ V))

    return jnp.abs(up / down) ** 2


def calculate_expvals(unitaries):

    expvals = []
    for expectation in specq.default_expectation_values:
        expvals_time = []
        for time_step in unitaries:
            exp = calculate_exp(
                time_step,
                expectation.observable_matrix,
                expectation.initial_statevector,
            )
            expvals_time.append(exp)

        expvals.append(expvals_time)

    return jnp.array(expvals)


def train_model(
    key: jnp.ndarray,
    model: nn.Module,
    optimiser: optax.GradientTransformation,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_epoch: int,
    progress_bar: bool = True,
    lr_transformer: Union[optax.GradientTransformationExtraArgs, None] = None,
    callback: Union[CallbackFn, None] = None,
):

    model_key, dropout_key = jax.random.split(key)

    train_step, test_step, model_params, opt_state, lr_transformer_state = (
        create_train_step(
            key=model_key,
            model=model,
            optimiser=optimiser,
            lr_transformer=lr_transformer,
            input_shape=next(iter(train_dataloader))["x0"].shape,
        )
    )

    model_params, opt_state, history = with_validation_train(
        train_dataloader,
        val_dataloader,
        train_step,
        test_step,
        model_params,
        opt_state,
        dropout_key,
        num_epochs=num_epoch,
        force_tty=progress_bar,
        lr_transformer=lr_transformer,
        lr_transformer_state=lr_transformer_state,
        callback=callback,
    )
    return model_params, opt_state, history


from .data import prepare_dataset, save_model, load_model
from enum import Enum
from .model import BasicBlackBox, MLPBlackBox
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import Searcher
import tempfile
from ray import tune, train
from .utils.helper import rotating_transmon_hamiltonian


def default_trainable(
    pulse_sequence: JaxBasedPulseSequence,
    with_dropout: bool = False,
):
    def trainable(
        config: dict,
        pulse_parameters: jnp.ndarray,
        unitaries: jnp.ndarray,
        expectations: jnp.ndarray,
        model_key: jnp.ndarray,
        train_key: jnp.ndarray,
    ):

        FEATURE_SIZE = config["feature_size"]
        HIDDEN_LAYER_1_1 = config["hidden_layer_1_1"]
        HIDDEN_LAYER_1_2 = config["hidden_layer_1_2"]
        HIDDEN_LAYER_2_1 = config["hidden_layer_2_1"]
        HIDDEN_LAYER_2_2 = config["hidden_layer_2_2"]

        model_config = {
            "feature_size": FEATURE_SIZE,
            "hidden_sizes_1": [HIDDEN_LAYER_1_1, HIDDEN_LAYER_1_2],
            "hidden_sizes_2": [HIDDEN_LAYER_2_1, HIDDEN_LAYER_2_2],
        }

        train_dataloader, val_dataloader = prepare_dataset(
            pulse_parameters=pulse_parameters,
            unitaries=unitaries,
            expectation_values=expectations,
            key=model_key,
        )

        lr_scheduler = optax.warmup_cosine_decay_schedule(
            init_value=1e-6,
            peak_value=1e-2,
            warmup_steps=1000,
            decay_steps=10000,
            end_value=1e-6,
        )

        optimiser = optax.adam(lr_scheduler)

        lr_transformer = default_adaptive_lr_transform(
            PATIENCE=20,
            COOLDOWN=0,
            FACTOR=0.5,
            RTOL=1e-4,
            ACCUMULATION_SIZE=100,
        )

        if not with_dropout:
            model: nn.Module = BasicBlackBox(**model_config)
        else:
            model = MLPBlackBox(**model_config)

        def callback(
            model_params: VariableDict,
            opt_state: optax.OptState,
            history: list[HistoryEntry],
        ) -> None:

            # Checkpoint the model
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = save_model(
                    path=str(tmpdir),
                    experiment_identifier="test",
                    pulse_sequence=pulse_sequence,
                    hamiltonian=rotating_transmon_hamiltonian,
                    model_config=model_config,
                    model_params=model_params,
                    history=history,
                    with_auto_datetime=False,
                )

                # Report the loss and val_loss to tune
                train.report(
                    metrics={
                        "val_loss": history[-1].val_loss,
                        "loss": history[-1].loss,
                    },
                    checkpoint=train.Checkpoint.from_directory(tmpdir),
                )

            return None

        model_opt_params, opt_state, history = train_model(
            key=train_key,
            model=model,
            optimiser=optimiser,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epoch=1500,
            lr_transformer=lr_transformer,
            callback=callback, # type: ignore
            progress_bar=False,
        )

        return {
            "val_loss": history[-1].val_loss,
            "loss": history[-1].loss,
        }

    return trainable


class SearchAlgo(Enum):
    HYPEROPT = "hyperopt"
    OPTUNA = "optuna"


def hypertuner(
    trainable: Callable,
    pulse_parameters: jnp.ndarray,
    unitaries: jnp.ndarray,
    expectations: jnp.ndarray,
    model_key: jnp.ndarray,
    train_key: jnp.ndarray,
    num_samples: int = 100,
    search_algo: SearchAlgo = SearchAlgo.HYPEROPT,
):

    search_space = {
        "feature_size": tune.randint(5, 10),
        "hidden_layer_1_1": tune.randint(5, 50),
        "hidden_layer_1_2": tune.randint(5, 50),
        "hidden_layer_2_1": tune.randint(5, 50),
        "hidden_layer_2_2": tune.randint(5, 50),
    }

    current_best_params = [
        {
            "feature_size": 5,
            "hidden_layer_1_1": 10,
            "hidden_layer_1_2": 20,
            "hidden_layer_2_1": 10,
            "hidden_layer_2_2": 20,
        }
    ]

    if search_algo == SearchAlgo.HYPEROPT:
        search_algo_instance: Searcher = HyperOptSearch(
            metric="val_loss", mode="min", points_to_evaluate=current_best_params
        )
    elif search_algo == SearchAlgo.OPTUNA:
        search_algo_instance = OptunaSearch(metric="val_loss", mode="min")

    tuner = tune.Tuner(
        tune.with_parameters(
            trainable,
            pulse_parameters=pulse_parameters,
            unitaries=unitaries,
            expectations=expectations,
            model_key=model_key,
            train_key=train_key,
        ),
        tune_config=tune.TuneConfig(
            search_alg=search_algo_instance,
            metric="val_loss",
            mode="min",
            num_samples=num_samples,
        ),
        param_space=search_space,
    )

    results = tuner.fit()

    return results


def get_best_hypertuner_results(results):
    with results.get_best_result(
        metric="val_loss", mode="min"
    ).checkpoint.as_directory() as checkpoint_dir:
        model_state, hist, data_config = load_model(str(checkpoint_dir))
    return model_state, hist, data_config
