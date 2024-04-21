import jax.numpy as jnp
import specq_dev.specq.shared as specq  # type: ignore
import pennylane as qml  # type: ignore
from typing import Callable
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from jaxtyping import Array, Complex, Float
from flax import linen as nn
import jax
import optax  # type: ignore
import matplotlib.pyplot as plt
from specq_dev.specq.jax import JaxBasedPulseSequence  # type: ignore
import pandas as pd
from alive_progress import alive_bar # type: ignore
from jaxopt import ProjectedGradient # type: ignore
from jaxopt.projection import projection_box # type: ignore

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


def loss(
    params,
    pulse_parameters: Float[Array, "batch num_pulses num_features"],  # noqa: F722
    unitaries: Complex[Array, "batch dim dim"],  # noqa: F722
    expectation_values: Complex[Array, "batch num_expectations"],  # noqa: F722
    model: nn.Module,
    evaluate_expectation_values: list[
        specq.ExpectationValue
    ] = specq.default_expectation_values,
):
    # Predict Vo for each pauli operator from paluse parameters
    Wos_params = model.apply(params, pulse_parameters)

    predict_expectation_values = []

    # Calculate expectation values for all cases
    for idx, exp_case in enumerate(evaluate_expectation_values):
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

    return jnp.mean(
        batch_mse(expectation_values, jnp.array(predict_expectation_values).T)
    )

def create_train_step(
    key: jnp.ndarray,
    model: nn.Module,
    optimiser: optax.GradientTransformation,
    loss_fn: Callable[
        [
            Array,
            Float[Array, "batch num_pulses num_features"],  # noqa: F722
            Complex[Array, "batch dim dim"],  # noqa: F722
            Complex[Array, "batch num_expectations"],  # noqa: F722
        ],
        Float[Array, "1"],
    ],
    input_shape: Array,
):
    params = model.init(
        key,
        jnp.ones(input_shape, jnp.float32),
    )  # dummy key just as example input
    opt_state = optimiser.init(params)

    @jax.jit
    def train_step(
        params,
        opt_state: optax.OptState,
        pulse_parameters: Float[Array, "batch num_pulses num_features"],  # noqa: F722
        unitaries: Complex[Array, "batch dim dim"],  # noqa: F722
        expectations: Complex[Array, "batch num_expectations"],  # noqa: F722
    ):
        loss, grads = jax.value_and_grad(loss_fn)(
            params, pulse_parameters, unitaries, expectations
        )

        updates, opt_state = optimiser.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss

    @jax.jit
    def test_step(
        params,
        pulse_parameters: Float[Array, "batch num_pulses num_features"],  # noqa: F722
        unitaries: Complex[Array, "batch dim dim"],  # noqa: F722
        expectations: Complex[Array, "batch num_expectations"],  # noqa: F722
    ):
        loss = loss_fn(params, pulse_parameters, unitaries, expectations)
        return loss

    return train_step, test_step, params, opt_state


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

    D = jnp.array([[lambda_1, 0], [0, lambda_2]])

    return Q @ D @ Q.T.conj()


signal = lambda params, t, phase, qubit_freq, total_time_ns: jnp.real(
    jnp.exp(1j * (2 * jnp.pi * qubit_freq * t + phase))
    * qml.pulse.pwc((0, total_time_ns))(params, t)
)

# The params are [amplitude, phase]
signal_with_phase = lambda params, t, qubit_freq, total_time_ns: jnp.real(
    jnp.exp(1j * (2 * jnp.pi * qubit_freq * t + params[1]))
    * qml.pulse.pwc((0, total_time_ns))(params[0], t)
)


def duffling_oscillator_hamiltonian(
    qubit_info: specq.QubitInformationV3,
    signal: Callable,
) -> qml.pulse.parametrized_hamiltonian.ParametrizedHamiltonian:
    static_hamiltonian = (
        2
        * jnp.pi
        * qubit_info.frequency
        * qml.jordan_wigner(qml.FermiC(0) * qml.FermiA(0))
    )
    drive_hamiltonian = (
        2
        * jnp.pi
        * qubit_info.drive_strength
        * qml.jordan_wigner(qml.FermiC(0) + qml.FermiA(0))
    )

    return static_hamiltonian + signal * drive_hamiltonian


def rotating_duffling_oscillator_hamiltonian(
    qubit_info: specq.QubitInformationV3, signal: Callable
) -> qml.pulse.parametrized_hamiltonian.ParametrizedHamiltonian:
    a0 = 2 * jnp.pi * qubit_info.frequency
    a1 = 2 * jnp.pi * qubit_info.drive_strength

    f3 = lambda params, t: a1 * signal(params, t)  # * ((a0 * t / 2)**2)
    f_sigma_x = lambda params, t: f3(params, t) * jnp.cos(a0 * t)
    f_sigma_y = lambda params, t: f3(params, t) * jnp.sin(a0 * t)

    return f_sigma_x * qml.PauliX(0) + f_sigma_y * qml.PauliY(0)


def transmon_hamiltonian(
    qubit_info: specq.QubitInformationV3, waveform: Callable
) -> qml.pulse.parametrized_hamiltonian.ParametrizedHamiltonian:
    H_0 = qml.pulse.transmon_interaction(
        qubit_freq=[qubit_info.frequency], connections=[], coupling=[], wires=[0]
    )
    H_d0 = qml.pulse.transmon_drive(
        waveform, 0, qubit_info.frequency, 0
    )  # NOTE: f2 must be real function.

    return H_0 + H_d0


def rotating_transmon_hamiltonian(
    qubit_info: specq.QubitInformationV3, signal: Callable
):

    a0 = 2 * jnp.pi * qubit_info.frequency
    a1 = 2 * jnp.pi * qubit_info.drive_strength

    f3 = lambda params, t: a1 * signal(params, t)
    f_sigma_x = lambda params, t: f3(params, t) * jnp.cos(a0 * t)
    f_sigma_y = lambda params, t: -1 * f3(params, t) * jnp.sin(a0 * t) / 2  # NOTE: WHY?

    return f_sigma_x * qml.PauliX(0) + f_sigma_y * qml.PauliY(0)


def whitebox(
    params: jnp.ndarray,
    t: jnp.ndarray,
    H: qml.pulse.parametrized_hamiltonian.ParametrizedHamiltonian,
    num_parameterized: int,
):
    # Evolve under the Hamiltonian
    unitary = qml.evolve(H)([params] * num_parameterized, t, return_intermediate=True)
    # Return the unitary
    return qml.matrix(unitary)


def get_simulator(
    qubit_info: specq.QubitInformationV3,
    t_eval: jnp.ndarray,
    hamiltonian: Callable[
        [specq.QubitInformationV3, Callable],
        qml.pulse.parametrized_hamiltonian.ParametrizedHamiltonian,
    ] = rotating_duffling_oscillator_hamiltonian,
):
    fixed_freq_signal = lambda params, t: signal(
        params,
        t,
        0,
        qubit_info.frequency,
        t_eval[-1],
    )

    H = hamiltonian(qubit_info, fixed_freq_signal)
    num_parameterized = len(H.coeffs_parametrized)

    simulator = lambda params: whitebox(params, t_eval, H, num_parameterized)

    return simulator


class SpecQDataset(Dataset):
    def __init__(
        self,
        pulse_parameters: np.ndarray,
        unitaries: np.ndarray,
        expectation_values: np.ndarray,
    ):
        self.pulse_parameters = torch.from_numpy(pulse_parameters)
        self.unitaries = torch.from_numpy(unitaries)
        self.expectation_values = torch.from_numpy(expectation_values)

    def __len__(self):
        return self.pulse_parameters.shape[0]

    def __getitem__(self, idx):
        return {
            # Pulse parameters
            "x0": self.pulse_parameters[idx],
            # Unitaries
            "x1": self.unitaries[idx],
            # Expectation values
            "y": self.expectation_values[idx],
        }


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


def plot_expvals(expvals):
    fig, axes = plt.subplots(3, 6, figsize=(20, 10), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flatten()):
        ax.plot(expvals[i])
        ax.set_title(
            f"{specq.default_expectation_values[i].initial_state}/{specq.default_expectation_values[i].observable}"
        )

    # Set the ylim to (-1.05, 1.05)
    for ax in axes.flatten():
        ax.set_ylim(-1.05, 1.05)

    # Set title for the figure
    fig.suptitle("Expectation values for the unitaries")

    plt.tight_layout()
    plt.show()


X, Y, Z = (
    jnp.array(specq.Operator.from_label("X")),
    jnp.array(specq.Operator.from_label("Y")),
    jnp.array(specq.Operator.from_label("Z")),
)


def gate_loss(
    x,
    model: nn.Module,
    model_params,
    simulator,
    pulse_sequence: JaxBasedPulseSequence,
    target_unitary,
):
    fidelities_dict: dict[str, jnp.ndarray] = {}
    # x is the pulse parameters in the flattened form
    # pulse_params = array_to_params_list(x)
    pulse_params = pulse_sequence.array_to_list_of_params(x)

    # Get the waveforms
    waveforms = pulse_sequence.get_waveform(pulse_params)

    # Get the unitaries
    unitaries = simulator(waveforms)

    # Evaluate model to get the Wo parameters
    Wo_params = model.apply(model_params, jnp.expand_dims(x, 0))
    # squeeze the Wo_params
    Wo_params = jax.tree_map(lambda x: jnp.squeeze(x, 0), Wo_params)

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

def with_validation_train(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    train_step,
    test_step,
    model_params,
    opt_state,
    num_epochs=1250,
):

    history = []
    total_len = len(train_dataloader)

    NUM_EPOCHS = num_epochs

    with alive_bar(int(NUM_EPOCHS * total_len), force_tty=True) as bar:
        for epoch in range(NUM_EPOCHS):
            total_loss = 0.0
            for i, batch in enumerate(train_dataloader):

                _pulse_parameters = batch["x0"].numpy()
                _unitaries = batch["x1"].numpy()
                _expectations = batch["y"].numpy()

                model_params, opt_state, loss = train_step(
                    model_params, opt_state, _pulse_parameters, _unitaries, _expectations
                )

                history.append(
                    {
                        "epoch": epoch,
                        "step": i,
                        "loss": float(loss),
                        "global_step": epoch * total_len + i,
                        "val_loss": None,
                    }
                )

                total_loss += loss

                bar()

            # Validation
            val_loss = 0.0
            for i, batch in enumerate(val_dataloader):

                _pulse_parameters = batch["x0"].numpy()
                _unitaries = batch["x1"].numpy()
                _expectations = batch["y"].numpy()

                val_loss += test_step(model_params, _pulse_parameters, _unitaries, _expectations)

            history[-1]["val_loss"] = float(val_loss / len(val_dataloader))


    return model_params, opt_state, history

def plot_history(history: list):

    hist_df = pd.DataFrame(history)
    train = hist_df[["global_step", "loss"]].values

    train_x = train[:, 0]
    train_y = train[:, 1]

    validate = hist_df[["global_step", "val_loss"]].replace(0, jnp.nan).dropna().values

    validate_x = validate[:, 0]
    validate_y = validate[:, 1]
    # The second plot has height ratio 2
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharex=True)

    # The first plot is the training loss and the validation loss
    ax.plot(train_x, train_y, label="train_loss")
    ax.plot(validate_x, validate_y, label="val_loss")
    ax.set_yscale("log")

    # plot the horizontal line [1e-3, 1e-2]
    ax.axhline(1e-3, color="red", linestyle="--")
    ax.axhline(1e-2, color="red", linestyle="--")

    ax.legend()

    fig.tight_layout()

    return fig, ax
    
def optimize(
    x0,
    lower,
    upper,
    fun
):

    pg = ProjectedGradient(fun=fun, projection=projection_box)
    opt_params, state = pg.run(
        jnp.array(x0),
        hyperparams_proj=(lower, upper)
    )

    return opt_params, state

def evalulate_model_to_pauli(model, model_params, pulse_params):

    Wo_params = model.apply(model_params, pulse_params)
    fidelities = {}
    for pauli_str, pauli_op in zip(["X", "Y", "Z"], [X, Y, Z]):
        Wos = jax.vmap(Wo_2_level, in_axes=(0, 0))(
            Wo_params[pauli_str]["U"], Wo_params[pauli_str]["D"]
        )
        _fidel = jax.vmap(gate_fidelity, in_axes=(0, None))(Wos, pauli_op)
        fidelities[pauli_str] = _fidel

    return fidelities