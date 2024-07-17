import os
import jax
from typing import Callable, Sequence
from flax import linen as nn
import jax.numpy as jnp
import typing
from flax.typing import VariableDict
import pandas as pd  # type: ignore
import json
from dataclasses import dataclass
import optax  # type: ignore
from torch.utils.data import DataLoader
from jaxtyping import Array, Complex, Float
from alive_progress import alive_bar  # type: ignore
from .typing import HistoryEntry, CallbackFn, TrainStepFn, TestStepFn
from .pulse import BasePulseSequence
from .physics import (
    SignalParameters,
    SimulatorFnV2,
    avg_gate_fidelity_from_superop,
    to_superop,
)
from .data import DataConfig, generate_path_with_datetime, ExpectationValue
from .constant import (
    X,
    Y,
    Z,
    default_expectation_values_order as default_expectation_values,
)


@dataclass
class ModelState:
    model_config: dict
    model_params: VariableDict

    def save(self, path: str):
        # turn the dataclass into a dictionary
        model_params = jax.tree.map(lambda x: x.tolist(), self.model_params)

        os.makedirs(path, exist_ok=True)

        with open(f"{path}/model_config.json", "w") as f:
            json.dump(self.model_config, f, indent=4)

        with open(f"{path}/model_params.json", "w") as f:
            json.dump(model_params, f, indent=4)

    @classmethod
    def load(cls, path: str):
        with open(f"{path}/model_config.json", "r") as f:
            model_config = json.load(f)

        with open(f"{path}/model_params.json", "r") as f:
            model_params = json.load(f)

        # Define is_leaf function to check if a node is a leaf
        # If the node is a leaf, convert it to a jnp.array
        # The function will check if object is list, if list then convert to jnp.array
        def is_leaf(x):
            return isinstance(x, list)

        # Apply the inverse of the tree.map function
        model_params = jax.tree.map(
            lambda x: jnp.array(x), model_params, is_leaf=is_leaf
        )

        return cls(model_config, model_params)


def save_model(
    path: str,
    experiment_identifier: str,
    pulse_sequence: BasePulseSequence,
    hamiltonian: typing.Union[str, typing.Callable],
    model_config: dict,
    model_params: VariableDict,
    history: list[HistoryEntry],
    with_auto_datetime: bool = True,
):

    _path = generate_path_with_datetime(path) if with_auto_datetime else path

    os.makedirs(_path, exist_ok=True)

    model_state = ModelState(
        model_config=model_config,
        model_params=model_params,
    )

    model_state.save(_path + "/model_state")

    # Save the history
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(_path + "/history.csv", index=False)

    # Save the data config
    data_config = DataConfig(
        EXPERIMENT_IDENTIFIER=experiment_identifier,
        hamiltonian=(
            hamiltonian if isinstance(hamiltonian, str) else hamiltonian.__name__
        ),
        pulse_sequence=pulse_sequence.to_dict(),
    )

    data_config.to_file(_path)

    return _path


def load_model(path: str, skip_history: bool = False):
    model_state = ModelState.load(path + "/model_state")
    if skip_history:
        hist_df = None
    else:
        hist_df = pd.read_csv(path + "/history.csv").to_dict(orient="records")
    data_config = DataConfig.from_file(path)

    return (
        model_state,
        hist_df,
        data_config,
    )


class BasicBlackBox(nn.Module):
    feature_size: int
    hidden_sizes_1: Sequence[int] = (20, 10)
    hidden_sizes_2: Sequence[int] = (20, 10)
    pauli_operators: Sequence[str] = ("X", "Y", "Z")

    NUM_UNITARY_PARAMS: int = 3
    NUM_DIAGONAL_PARAMS: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool):
        x = nn.Dense(features=self.feature_size)(x)
        # Flatten the input
        x = x.reshape((x.shape[0], -1))
        # Apply a activation function
        x = nn.relu(x)
        # Dropout
        # x = nn.Dropout(0.2)(x)
        # Apply a dense layer for each hidden size
        for hidden_size in self.hidden_sizes_1:
            x = nn.Dense(features=hidden_size)(x)
            x = nn.relu(x)

        Wos_params: dict[str, dict[str, jnp.ndarray]] = dict()
        for op in self.pauli_operators:
            # Sub hidden layer
            for hidden_size in self.hidden_sizes_2:
                _x = nn.Dense(features=hidden_size)(x)
                _x = nn.relu(_x)

            Wos_params[op] = dict()
            # For the unitary part, we use a dense layer with 3 features
            unitary_params = nn.Dense(features=self.NUM_UNITARY_PARAMS)(_x)
            # Apply sigmoid to this layer
            unitary_params = 2 * jnp.pi * nn.sigmoid(unitary_params)
            # For the diagonal part, we use a dense layer with 1 feature
            diag_params = nn.Dense(features=self.NUM_DIAGONAL_PARAMS)(_x)
            # Apply the activation function
            diag_params = nn.tanh(diag_params)

            Wos_params[op] = {
                "U": unitary_params,
                "D": diag_params,
            }

        return Wos_params


class MLPBlackBox(nn.Module):
    feature_size: int
    hidden_sizes_1: Sequence[int] = (20, 10)
    hidden_sizes_2: Sequence[int] = (20, 10)
    pauli_operators: Sequence[str] = ("X", "Y", "Z")

    NUM_UNITARY_PARAMS: int = 3
    NUM_DIAGONAL_PARAMS: int = 2
    DROPOUT_RATE: float = 0.2

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool):
        x = nn.Dense(features=self.feature_size)(x)
        # Flatten the input
        x = x.reshape((x.shape[0], -1))
        # Apply a activation function
        x = nn.relu(x)
        # Dropout
        x = nn.Dropout(rate=self.DROPOUT_RATE, deterministic=not training)(x)
        # Apply a dense layer for each hidden size
        for hidden_size in self.hidden_sizes_1:
            x = nn.Dense(features=hidden_size)(x)
            x = nn.relu(x)
            # Dropout
            x = nn.Dropout(rate=self.DROPOUT_RATE, deterministic=not training)(x)

        Wos_params: dict[str, dict[str, jnp.ndarray]] = dict()
        for op in self.pauli_operators:
            # Sub hidden layer
            for hidden_size in self.hidden_sizes_2:
                _x = nn.Dense(features=hidden_size)(x)
                _x = nn.relu(_x)
                # Dropout
                _x = nn.Dropout(rate=self.DROPOUT_RATE, deterministic=not training)(_x)

            Wos_params[op] = dict()
            # For the unitary part, we use a dense layer with 3 features
            unitary_params = nn.Dense(features=self.NUM_UNITARY_PARAMS)(_x)
            # Apply sigmoid to this layer
            unitary_params = 2 * jnp.pi * nn.sigmoid(unitary_params)
            # For the diagonal part, we use a dense layer with 1 feature
            diag_params = nn.Dense(features=self.NUM_DIAGONAL_PARAMS)(_x)
            # Apply the activation function
            diag_params = nn.tanh(diag_params)

            Wos_params[op] = {
                "U": unitary_params,
                "D": diag_params,
            }

        return Wos_params


class DirectWoMLPBlackBox(nn.Module):
    feature_size: int
    hidden_sizes_1: Sequence[int] = (20, 10)
    hidden_sizes_2: Sequence[int] = (20, 10)
    pauli_operators: Sequence[str] = ("X", "Y", "Z")

    dims: int = 2
    DROPOUT_RATE: float = 0.2

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool):
        x = nn.Dense(features=self.feature_size)(x)
        # Flatten the input
        x = x.reshape((x.shape[0], -1))
        # Apply a activation function
        x = nn.relu(x)
        # Dropout
        x = nn.Dropout(rate=self.DROPOUT_RATE, deterministic=not training)(x)
        # Apply a dense layer for each hidden size
        for hidden_size in self.hidden_sizes_1:
            x = nn.Dense(features=hidden_size)(x)
            x = nn.relu(x)
            # Dropout
            x = nn.Dropout(rate=self.DROPOUT_RATE, deterministic=not training)(x)

        Wos: dict[str, jnp.ndarray] = dict()
        for op in self.pauli_operators:
            # Sub hidden layer
            for hidden_size in self.hidden_sizes_2:
                _x = nn.Dense(features=hidden_size)(x)
                _x = nn.relu(_x)
                # Dropout
                _x = nn.Dropout(rate=self.DROPOUT_RATE, deterministic=not training)(_x)

            # Hermitian matrix
            hermitian = nn.Dense(features=int(self.dims * self.dims))(_x)
            hermitian = hermitian.reshape((-1, self.dims, self.dims))
            # Make the matrix Hermitian
            hermitian = (hermitian + jnp.conj(hermitian.T)) / 2

            Wos[op] = hermitian

        return Wos


class ParallelBlackBox(nn.Module):
    hidden_sizes: Sequence[int] = (20, 10)
    pauli_operators: Sequence[str] = ("X", "Y", "Z")

    activation: Callable = nn.tanh
    NUM_UNITARY_PARAMS: int = 3
    NUM_DIAGONAL_PARAMS: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray):

        Wos_params: dict[str, dict[str, jnp.ndarray]] = dict()
        for op in self.pauli_operators:
            # Sub hidden layer
            for hidden_size in self.hidden_sizes:
                _x = nn.Dense(features=hidden_size)(x)
                _x = nn.relu(_x)

            Wos_params[op] = dict()
            # For the unitary part, we use a dense layer with 3 features
            unitary_params = nn.Dense(features=self.NUM_UNITARY_PARAMS)(_x)
            # Apply sigmoid to this layer
            unitary_params = 2 * jnp.pi * nn.sigmoid(unitary_params)
            # For the diagonal part, we use a dense layer with 1 feature
            diag_params = nn.Dense(features=self.NUM_DIAGONAL_PARAMS)(_x)
            # Apply the activation function
            diag_params = self.activation(diag_params)

            Wos_params[op] = {
                "U": unitary_params,
                "D": diag_params,
            }

        return Wos_params


class GRUBlackBox(nn.Module):

    @nn.compact
    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


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


def calculate_exp(
    unitary: jnp.ndarray, operator: jnp.ndarray, initial_state: jnp.ndarray
) -> jnp.ndarray:
    density_matrix = jnp.outer(initial_state, initial_state.T.conj())
    rho = jnp.matmul(unitary, jnp.matmul(density_matrix, unitary.T.conj()))
    temp = jnp.matmul(rho, operator)
    return jnp.real(jnp.trace(temp))


def calculate_expvals(unitaries):

    expvals = []
    for expectation in default_expectation_values:
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
    evaluate_expectation_values: list[ExpectationValue],
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
    evaluate_expectation_values: list[ExpectationValue] = default_expectation_values,
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
    evaluate_expectation_values: list[ExpectationValue] = default_expectation_values,
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
    lr_transformer: typing.Union[None, optax.GradientTransformationExtraArgs] = None,
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
    lr_transformer: typing.Union[None, optax.GradientTransformationExtraArgs] = None,
    lr_transformer_state: typing.Union[None, optax.OptState] = None,
    callback: typing.Union[CallbackFn, None] = None,
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
                        batch_loss=float(loss),
                        global_step=epoch * total_len + i,
                    )
                )

                total_loss += loss

                bar()

            epoch_loss = float(total_loss / total_len)
            history[-1].epoch_loss = epoch_loss

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


def train_model(
    key: jnp.ndarray,
    model: nn.Module,
    optimiser: optax.GradientTransformation,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_epoch: int,
    progress_bar: bool = True,
    lr_transformer: typing.Union[optax.GradientTransformationExtraArgs, None] = None,
    callback: typing.Union[CallbackFn, None] = None,
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


def gate_loss_v2(
    x: jnp.ndarray,
    model: nn.Module,
    model_params: VariableDict,
    simulator: SimulatorFnV2,
    # pulse_sequence: JaxBasedPulseSequence,
    array_to_list_of_params: Callable,
    target_unitary: jnp.ndarray,
):
    fidelities_dict: dict[str, jnp.ndarray] = {}
    # x is the pulse parameters in the flattened form
    # pulse_params = pulse_sequence.array_to_list_of_params(x)
    pulse_params = array_to_list_of_params(x)

    # Get the waveforms
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
        value = avg_gate_fidelity_from_superop(to_superop(Wo), to_superop(pauli_op))
        fidelities_dict[pauli_str] = value

        loss += (1 - value) ** 2

    gate_fi = avg_gate_fidelity_from_superop(
        to_superop(unitaries[-1]), to_superop(target_unitary)
    )
    fidelities_dict["gate"] = gate_fi

    # Calculate the loss
    loss += (1 - gate_fi) ** 2

    return loss


def evalulate_model_to_pauli_avg_gate_fidelity(
    model: nn.Module, model_params, pulse_params
):

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
        _fidel = jax.vmap(avg_gate_fidelity_from_superop, in_axes=(0, None))(
            to_superop(Wos), to_superop(pauli_op)
        )
        fidelities[pauli_str] = _fidel

    return fidelities
