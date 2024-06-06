import jax.numpy as jnp
from specq_dev import qiskit as qk, shared  # type: ignore
from ..simulator import HamiltonianTerm, ChannelID, TermType, tensor, N, a, a_dag
import logging
from ..data import pulse_reader
from ..pennylane.simulator import (
    get_simulator,
    rotating_duffling_oscillator_hamiltonian,
)
import jax
import numpy as np
import flax.linen as nn
from flax.typing import VariableDict
from typing import Protocol
from specq_dev.jax import JaxBasedPulseSequence  # type: ignore
from ..core import X, Y, Z, Wo_2_level, gate_fidelity


def gen_hamiltonian_from(backend_properties: qk.IBMQDeviceProperties, dims: int = 2):
    # https://qiskit-extensions.github.io/qiskit-dynamics/stubs/qiskit_dynamics.backend.parse_backend_hamiltonian_dict.html
    # Get the qubits coupling map
    qubit_idxs = [qubit.qubit_idx for qubit in backend_properties.qubit_informations]
    # Get the edges of the coupling map
    edge_reindex = list(
        backend_properties.backend_instance.coupling_map.reduce(qubit_idxs).get_edges()
    )
    # Map the edges to the qubit indices
    edges = [(qubit_idxs[edge[0]], qubit_idxs[edge[1]]) for edge in edge_reindex]
    # Get the coupling constants
    coupling_constants = []
    for i, j in edges:
        # Get the coupling constant
        coupling_const = backend_properties.hamiltonian_data["vars"].get(
            f"jq{i}q{j}", 0
        )
        # Add the coupling constant to the list
        coupling_constants.append(
            (
                (i, j),
                coupling_const / (2 * jnp.pi),
            )
        )

    num_qubits = len(qubit_idxs)

    operators: dict[str, HamiltonianTerm] = {}

    for idx, qubit in enumerate(backend_properties.qubit_informations):

        # The static Hamiltonian terms
        static_i = 2 * jnp.pi * qubit.frequency * (jnp.eye(dims) - 2 * N(dims)) / 2

        operators[ChannelID(qubit_idx=qubit.qubit_idx, type=TermType.STATIC).hash()] = (
            HamiltonianTerm(
                qubit_idx=qubit.qubit_idx,
                type=TermType.STATIC,
                controlable=False,
                operator=tensor(static_i, idx, num_qubits),
            )
        )

        # The anharmonicity term
        anhar_i = 2 * jnp.pi * qubit.anharmonicity * (N(dims) @ N(dims) - N(dims)) / 2

        operators[
            ChannelID(qubit_idx=qubit.qubit_idx, type=TermType.ANHAMONIC).hash()
        ] = HamiltonianTerm(
            qubit_idx=qubit.qubit_idx,
            type=TermType.ANHAMONIC,
            controlable=False,
            operator=tensor(anhar_i, idx, num_qubits),
        )

        # The drive terms
        drive_i = 2 * jnp.pi * qubit.drive_strength * (a(dims) + a_dag(dims))

        operators[ChannelID(qubit_idx=qubit.qubit_idx, type=TermType.DRIVE).hash()] = (
            HamiltonianTerm(
                qubit_idx=qubit.qubit_idx,
                type=TermType.DRIVE,
                controlable=True,
                operator=tensor(drive_i, idx, num_qubits),
            )
        )

        # The control terms that drive with another qubit frequency
        control_i = 2 * jnp.pi * qubit.drive_strength * (a(dims) + a_dag(dims))

        operators[
            ChannelID(qubit_idx=qubit.qubit_idx, type=TermType.CONTROL).hash()
        ] = HamiltonianTerm(
            qubit_idx=qubit.qubit_idx,
            type=TermType.CONTROL,
            controlable=True,
            operator=tensor(control_i, idx, num_qubits),
        )

    for coupling in coupling_constants:
        # Add the coupling constant to the Hamiltonian
        c_1 = tensor(a(dims), coupling[0][0], num_qubits) @ tensor(
            a_dag(dims), coupling[0][1], num_qubits
        )
        c_2 = tensor(a_dag(dims), coupling[0][0], num_qubits) @ tensor(
            a(dims), coupling[0][1], num_qubits
        )
        coupling_ij = 2 * jnp.pi * coupling[1] * (c_1 + c_2)

        operators[
            ChannelID(
                qubit_idx=(coupling[0][0], coupling[0][1]), type=TermType.COUPLING
            ).hash()
        ] = HamiltonianTerm(
            qubit_idx=(coupling[0][0], coupling[0][1]),
            type=TermType.COUPLING,
            controlable=False,
            operator=coupling_ij,
        )

    return operators


def load_data(
    path: str,
    hamiltonian=rotating_duffling_oscillator_hamiltonian,
):
    # Load the data from the experiment
    exp_data = shared.ExperimentData.from_folder(path)

    logging.info(f"Loaded data from {exp_data.experiment_config.EXPERIMENT_IDENTIFIER}")

    # Setup the simulator
    dt = exp_data.experiment_config.device_cycle_time_ns
    qubit_info = exp_data.experiment_config.qubits[0]
    # pulse_sequence = get_pulse_sequence()
    pulse_sequence = pulse_reader(path)
    t_eval = jnp.linspace(
        0, pulse_sequence.pulse_length_dt * dt, pulse_sequence.pulse_length_dt
    )
    simulator = get_simulator(
        qubit_info=qubit_info, t_eval=t_eval, hamiltonian=hamiltonian
    )

    # Get the waveforms for each pulse parameters to get the unitaries
    _waveforms = []
    for pulse_params in exp_data.get_parameters_dict_list():
        _waveforms.append(pulse_sequence.get_waveform(pulse_params))

    waveforms = jnp.array(_waveforms)

    logging.info(
        f"Prepared the waveforms for the experiment {exp_data.experiment_config.EXPERIMENT_IDENTIFIER}"
    )

    # Check if the unitaries are already calculated
    # if os.path.exists(f"{path}/unitaries.npy"):
    #     unitaries = np.load(f"{path}/unitaries.npy")
    #     logging.info(
    #         f"Loaded the unitaries for the experiment {exp_data.experiment_config.EXPERIMENT_IDENTIFIER}"
    #     )
    # else:

    # jit the simulator
    jitted_simulator = jax.jit(simulator)
    # batch the simulator
    batched_simulator = jax.vmap(jitted_simulator, in_axes=(0))
    # Get the unitaries
    unitaries = batched_simulator(waveforms)

    logging.info(
        f"Got the unitaries for the experiment {exp_data.experiment_config.EXPERIMENT_IDENTIFIER}"
    )

    # Get the final unitaries
    unitaries = np.array(unitaries[:, -1, :, :])
    # Save the unitaries
    # np.save(f"{path}/unitaries.npy", unitaries)

    # Get the expectation values from the experiment
    expectations = exp_data.get_expectation_values()
    # Get the pulse parameters
    pulse_parameters = exp_data.parameters

    logging.info(
        f"Finished preparing the data for the experiment {exp_data.experiment_config.EXPERIMENT_IDENTIFIER}"
    )

    return (
        exp_data,
        pulse_parameters,
        unitaries,
        expectations,
        pulse_sequence,
        simulator,
    )


class SimulatorFnV1(Protocol):
    def __call__(self, waveforms: jnp.ndarray) -> jnp.ndarray: ...


def gate_loss(
    x: jnp.ndarray,
    model: nn.Module,
    model_params: VariableDict,
    simulator: SimulatorFnV1,
    pulse_sequence: JaxBasedPulseSequence,
    target_unitary: jnp.ndarray,
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
