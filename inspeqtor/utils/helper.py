import jax
import jax.numpy as jnp
from functools import partial
from pathlib import Path
from os import PathLike
import numpy as np
import logging
import typing

from enum import Enum
from functools import partial
import pandas as pd  # type: ignore

from ..model import calculate_exp
from .predefined import (
    MultiDragPulseV2,
    JaxDragPulse,
    DragPulse,
    rotating_transmon_hamiltonian,
    get_mock_prefined_exp_v1,
    default_expectation_values_order,
)
from ..pulse import (
    JaxBasedPulseSequence,
    construct_pulse_sequence_reader,
    list_of_params_to_array,
)
from ..data import ExperimentData, State, make_row
from ..physics import signal_func_v3, solver, SignalParameters
from ..decorator import not_yet_tested


def retrieve_keys(MASTER_KEY_SEED: int):
    master_key = jax.random.PRNGKey(MASTER_KEY_SEED)
    pulse_random_split_key, random_split_key, model_key, dropout_key, gate_optim_key = (
        jax.random.split(master_key, 5)
    )
    return (
        pulse_random_split_key,  # Random pulse sequence key
        random_split_key,  # Random split dataset key
        model_key,  # Model key
        dropout_key,  # Dropout key
        gate_optim_key,  # Gate optimization key
    )


def check_diagonal_matrix(matrix):
    return jnp.count_nonzero(matrix - jnp.diag(jnp.diagonal(matrix))) == 0


pulse_reader = construct_pulse_sequence_reader(
    pulses=[DragPulse, JaxDragPulse, MultiDragPulseV2]
)


def prepare_data(
    exp_data: ExperimentData,
    pulse_sequence: JaxBasedPulseSequence,
) -> tuple[
    ExperimentData,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    JaxBasedPulseSequence,
    typing.Callable,
]:
    dt = exp_data.experiment_config.device_cycle_time_ns
    qubit_info = exp_data.experiment_config.qubits[0]

    logging.info(f"Loaded data from {exp_data.experiment_config.EXPERIMENT_IDENTIFIER}")

    t_eval = jnp.linspace(
        0, pulse_sequence.pulse_length_dt * dt, pulse_sequence.pulse_length_dt
    )

    hamiltonian = partial(
        rotating_transmon_hamiltonian,
        qubit_info=qubit_info,
        signal=signal_func_v3(
            pulse_sequence.get_envelope,
            qubit_info.frequency,
            dt,
        ),
    )

    jiited_simulator = jax.jit(
        partial(
            solver,
            t_eval=t_eval,
            hamiltonian=hamiltonian,
            y0=jnp.eye(2, dtype=jnp.complex64),
            t0=0,
            t1=pulse_sequence.pulse_length_dt * dt,
        )
    )

    signal_params: list[SignalParameters] = []
    for pulse_params in exp_data.get_parameters_dict_list():
        signal_params.append(SignalParameters(pulse_params=pulse_params, phase=0))

    unitaries = []
    for signal_param in signal_params:
        unitaries.append(jiited_simulator(signal_param))

    np_unitaries = np.array(unitaries)[:, -1, :, :]

    logging.info(
        f"Finished preparing the data for the experiment {exp_data.experiment_config.EXPERIMENT_IDENTIFIER}"
    )

    pulse_parameters = exp_data.parameters
    expectations = exp_data.get_expectation_values()

    return (
        exp_data,
        pulse_parameters,
        np_unitaries,
        expectations,
        pulse_sequence,
        jiited_simulator,
    )


def load_data_from_path(
    path: PathLike,
) -> tuple[
    ExperimentData,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    JaxBasedPulseSequence,
    typing.Callable,
]:

    # Load the data from the experiment
    exp_data = ExperimentData.from_folder(path)
    pulse_sequence = pulse_reader(str(path))

    return prepare_data(exp_data, pulse_sequence)


class SimulationStrategy(Enum):
    IDEAL = "ideal"
    RANDOM = "random"
    SHOT = "shot"
    NOISY = "noisy"


def calculate_shots_expectation_value(
    key,
    initial_state: jnp.ndarray,
    unitary: jnp.ndarray,
    plus_projector: jnp.ndarray,
    shots: int,
):

    prob = jnp.trace(unitary @ initial_state @ unitary.conj().T @ plus_projector).real

    return jax.random.choice(
        key, jnp.array([1, -1]), shape=(shots,), p=jnp.array([prob, 1 - prob])
    ).mean()


def test_calculate_shots_expectation_value(
    key: jnp.ndarray,
):
    initial_state = State.from_label("+", dm=True)
    projector = State.from_label("0", dm=True)

    key, sample_key = jax.random.split(key)

    expval = calculate_shots_expectation_value(
        key=sample_key,
        initial_state=initial_state,
        # unitary=unitaries[0][-1],
        unitary=jnp.eye(2),
        plus_projector=projector,
        shots=3000,
    )


plus_projectors = {
    "X": State.from_label("+", dm=True),
    "Y": State.from_label("r", dm=True),
    "Z": State.from_label("0", dm=True),
}


@not_yet_tested
def generate_mock_experiment_data(
    key: jnp.ndarray,
    sample_size: int = 10,
    shots: int = 1000,
    strategy: SimulationStrategy = SimulationStrategy.RANDOM,
):

    qubit_info, pulse_sequence, config = get_mock_prefined_exp_v1(
        sample_size=sample_size, shots=shots
    )

    # Generate mock expectation value
    key, exp_key = jax.random.split(key)

    dt = config.device_cycle_time_ns
    t_eval = jnp.linspace(
        0, pulse_sequence.pulse_length_dt * dt, pulse_sequence.pulse_length_dt
    )

    hamiltonian = partial(
        rotating_transmon_hamiltonian,
        qubit_info=qubit_info,
        signal=signal_func_v3(
            pulse_sequence.get_envelope,
            qubit_info.frequency,
            dt,
        ),
    )

    jiited_simulator = jax.jit(
        partial(
            solver,
            t_eval=t_eval,
            hamiltonian=hamiltonian,
            y0=jnp.eye(2, dtype=jnp.complex64),
            t0=0,
            t1=pulse_sequence.pulse_length_dt * dt,
        )
    )

    SHOTS = config.shots

    rows = []
    # manual_pulse_params_array = []
    pulse_params_list = []
    signal_params_list: list[SignalParameters] = []
    parameter_structure = pulse_sequence.get_parameter_names()
    for sample_idx in range(config.sample_size):

        key, subkey = jax.random.split(key)
        pulse_params = pulse_sequence.sample_params(subkey)
        pulse_params_list.append(pulse_params)
        # manual_pulse_params_array.append(
        #     list_of_params_to_array(pulse_params, parameter_structure)
        # )

        signal_param = SignalParameters(pulse_params=pulse_params, phase=0)
        signal_params_list.append(signal_param)

    # Calculate the unitaries
    unitaries = []
    for signal_param in signal_params_list:
        unitary = jiited_simulator(signal_param)
        unitaries.append(unitary)

    # Calculate the expectation values depending on the strategy
    expectation_values: jnp.ndarray
    expvals = []
    final_unitaries = jnp.array(unitaries)[:, -1, :, :]

    assert final_unitaries.shape == (
        sample_size,
        2,
        2,
    ), f"Final unitaries shape is {final_unitaries.shape}"

    if strategy == SimulationStrategy.RANDOM:
        # Just random expectation values with key
        expectation_values = 2 * (
            jax.random.uniform(exp_key, shape=(config.sample_size, 18)) - (1 / 2)
        )
    elif strategy == SimulationStrategy.IDEAL:

        for exp in default_expectation_values_order:
            expval = jax.vmap(calculate_exp, in_axes=(0, None, None))(
                final_unitaries, exp.observable_matrix, exp.initial_statevector
            )

            expvals.append(expval)

        expectation_values = jnp.array(expvals).T

    elif strategy == SimulationStrategy.SHOT:

        for exp in default_expectation_values_order:
            key, sample_key = jax.random.split(key)
            sample_keys = jax.random.split(sample_key, num=final_unitaries.shape[0])

            expval = jax.vmap(
                calculate_shots_expectation_value, in_axes=(0, None, 0, None, None)
            )(
                sample_keys,
                exp.initial_density_matrix,
                final_unitaries,
                plus_projectors[exp.observable],
                SHOTS,
            )

            expvals.append(expval)

        expectation_values = jnp.array(expvals).T

    else:
        raise NotImplementedError

    assert expectation_values.shape == (
        sample_size,
        18,
    ), f"Expectation values shape is {expectation_values.shape}"

    for sample_idx in range(config.sample_size):

        for exp_idx, exp in enumerate(default_expectation_values_order):
            row = make_row(
                expectation_value=float(expectation_values[sample_idx, exp_idx]),
                initial_state=exp.initial_state,
                observable=exp.observable,
                parameters_list=pulse_params_list[sample_idx],
                parameters_id=sample_idx,
            )

            rows.append(row)

    df = pd.DataFrame(rows)

    # manual_pulse_params_array = jnp.array(manual_pulse_params_array)
    exp_data = ExperimentData(experiment_config=config, preprocess_data=df)

    return (
        exp_data,
        pulse_sequence,
        jnp.array(unitaries),
        signal_params_list,
        jiited_simulator,
    )


def theoretical_fidelity_of_amplitude_dampling_channel_with_itself(gamma):

    process_fidelity = 1 - gamma + (gamma**2) / 2

    return (2 * process_fidelity + 1) / (2 + 1)
