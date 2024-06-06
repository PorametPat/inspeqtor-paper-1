from flax import struct
import jax
import jax.numpy as jnp
from typing import Callable, Protocol, TypeVar
from specq_dev import shared  # type: ignore
from specq_dev.jax import JaxBasedPulseSequence  # type: ignore
import diffrax  # type: ignore
from enum import Enum
from dataclasses import dataclass
from specq_dev import qiskit as qk  # type: ignore


@struct.dataclass
class SignalParameters:
    pulse_params: list[shared.ParametersDictType]
    phase: float


class SimulatorFnV2(Protocol):
    def __call__(
        self,
        args: SignalParameters,
    ) -> jnp.ndarray: ...


class TermType(Enum):
    STATIC = 0
    ANHAMONIC = 1
    DRIVE = 2
    CONTROL = 3
    COUPLING = 4


@struct.dataclass
class HamiltonianTerm:
    qubit_idx: int | tuple[int, int]
    type: TermType
    controlable: bool
    operator: jnp.ndarray


@dataclass
class CouplingInformation:
    qubit_indices: tuple[int, int]
    coupling_strength: float


@dataclass
class ChannelID:
    qubit_idx: int | tuple[int, int]
    type: TermType

    def hash(self):
        if self.type == TermType.COUPLING:
            return f"ct/{self.type.value}/{self.qubit_idx[0]}/{self.qubit_idx[1]}"
        return f"ct/{self.type.value}/{self.qubit_idx}"

    @classmethod
    def from_hash(cls, hash: str):
        parts = hash.split("/")
        if parts[1] == "4":
            return cls(
                qubit_idx=(int(parts[2]), int(parts[3])), type=TermType(int(parts[1]))
            )
        return cls(qubit_idx=int(parts[2]), type=TermType(int(parts[1])))


def signal_func_v3(get_envelope: Callable, drive_frequency: float, dt: float):
    """Make the envelope function into signal with drive frequency

    Args:
        get_envelope (Callable): The envelope function in unit of dt
        drive_frequency (float): drive freuqency in unit of GHz
        dt (float): The dt provived will be used to convert envelope unit to ns,
                    set to 1 if the envelope function is already in unit of ns
    """

    def signal(pulse_parameters: SignalParameters, t: jnp.ndarray):
        return jnp.real(
            get_envelope(pulse_parameters.pulse_params)(t / dt)
            * jnp.exp(
                1j * ((2 * jnp.pi * drive_frequency * t) + pulse_parameters.phase)
            )
        )

    return signal


def signal_func_v4(
    pulse_sequence: JaxBasedPulseSequence, drive_frequency: float, dt: float
):
    """Make the envelope function into signal with drive frequency
       This implementation intended for signal that will evaluate with just array
       make it compatable with pennylane syntax.
       Where we assume that the last element of the array is the drive frequence of the signal.

    Args:
        get_envelope (Callable): The envelope function in unit of dt
        drive_frequency (float): drive freuqency in unit of GHz
        dt (float): The dt provived will be used to convert envelope unit to ns,
                    set to 1 if the envelope function is already in unit of ns
    """

    def signal(pulse_parameters: jnp.ndarray, t: jnp.ndarray):
        return jnp.real(
            pulse_sequence.get_envelope(
                pulse_sequence.array_to_list_of_params(pulse_parameters[:-1])
            )(t / dt)
            * jnp.exp(1j * ((2 * jnp.pi * drive_frequency * t) + pulse_parameters[-1]))
        )

    return signal


def normalizer(matrix: jnp.ndarray) -> jnp.ndarray:
    """Normalize the given matrix with QR decomposition and return matrix Q
       which is unitary

    Args:
        matrix (jnp.ndarray): The matrix to normalize to unitary matrix

    Returns:
        jnp.ndarray: The unitary matrix
    """
    return jnp.linalg.qr(matrix).Q  # type: ignore


HamiltonianArgs = TypeVar("HamiltonianArgs")


def simulator(
    args: HamiltonianArgs,
    t_eval: jnp.ndarray,
    hamiltonian: Callable,
    y0: jnp.ndarray,
    max_steps: int = int(2**16),
):

    # NOTE: Increase time_step to increase accuracy of solver,
    #       then you have to increase the max_steps too.
    # NOTE: Using just a basic solver
    def rhs(t: jnp.ndarray, y: jnp.ndarray, args: HamiltonianArgs):
        return -1j * hamiltonian(args, t) @ y

    term = diffrax.ODETerm(rhs)
    # solver = Dopri5()
    # solver = Dopri8()
    solver = diffrax.Tsit5()

    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=t_eval[0],
        t1=t_eval[-1],
        dt0=None,
        stepsize_controller=diffrax.PIDController(
            rtol=1e-7,
            atol=1e-7,
        ),
        y0=y0,
        args=args,
        saveat=diffrax.SaveAt(ts=t_eval),
        max_steps=max_steps,
    )

    # Normailized the solution
    normalized_solutions = jax.vmap(normalizer)(solution.ys)

    return normalized_solutions


def auto_rotating_frame_hamiltonian(
    hamiltonian: Callable[[HamiltonianArgs, jnp.ndarray], jnp.ndarray], frame: jnp.ndarray, explicit_deriv: bool = False
):
    """Implement the Hamiltonian in the rotating frame with
    H_I = U(t) @ H @ U^dagger(t) + i * U(t) @ dU^dagger(t)/dt

    Args:
        hamiltonian (Callable): The hamiltonian function
        frame (jnp.ndarray): The frame matrix
    """

    is_diagonal = False
    # Check if the frame is diagonal matrix
    if jnp.count_nonzero(frame - jnp.diag(jnp.diagonal(frame))) == 0:
        is_diagonal = True

    # Check if the jax_enable_x64 is True
    if not jax.config.read("jax_enable_x64") or is_diagonal:

        def frame_unitary(t: jnp.ndarray) -> jnp.ndarray:
            # NOTE: This is the same as the below, as we sure that the frame is diagonal
            return jnp.diag(jnp.exp(1j * jnp.diagonal(frame) * t))

    else:

        def frame_unitary(t: jnp.ndarray) -> jnp.ndarray:
            return jax.scipy.linalg.expm(1j * frame * t)

    def derivative_frame_unitary(t: jnp.ndarray) -> jnp.ndarray:
        # NOTE: Assume that the frame is time independent.
        return 1j * frame @ frame_unitary(t)

    def rotating_frame_hamiltonian_v0(args: HamiltonianArgs, t: jnp.ndarray):
        return frame_unitary(t) @ hamiltonian(args, t) @ jnp.transpose(
            jnp.conjugate(frame_unitary(t))
        ) + 1j * (
            derivative_frame_unitary(t) @ jnp.transpose(jnp.conjugate(frame_unitary(t)))
        )

    def rotating_frame_hamiltonian(args: HamiltonianArgs, t: jnp.ndarray):
        # NOTE: Assume that the product of derivative and conjugate of frame unitary is identity
        return (
            frame_unitary(t)
            @ hamiltonian(args, t)
            @ jnp.transpose(jnp.conjugate(frame_unitary(t)))
            - frame
        )

    return (
        rotating_frame_hamiltonian_v0 if explicit_deriv else rotating_frame_hamiltonian
    )


def a(dims: int):
    return jnp.diag(jnp.sqrt(jnp.arange(1, dims)), 1)


def a_dag(dims: int):
    return jnp.diag(jnp.sqrt(jnp.arange(1, dims)), -1)


def N(dims: int):
    return jnp.diag(jnp.arange(dims))


def tensor(operator: jnp.ndarray, position: int, total_qubits: int):

    return jnp.kron(
        jnp.eye(operator.shape[0] ** position),
        jnp.kron(operator, jnp.eye(operator.shape[0] ** (total_qubits - position - 1))),
    )


def get_coupling_strengths(backend_properties: qk.IBMQDeviceProperties):
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
            CouplingInformation(
                qubit_indices=(i, j),
                coupling_strength=coupling_const / (2 * jnp.pi),
            )
        )
    return coupling_constants


def gen_hamiltonian_from(
    qubit_informations: list[qk.QubitInformation],
    coupling_constants: list[CouplingInformation],
    dims: int = 2,
):

    num_qubits = len(qubit_informations)

    operators: dict[str, HamiltonianTerm] = {}

    for idx, qubit in enumerate(qubit_informations):

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
        c_1 = tensor(a(dims), coupling.qubit_indices[0], num_qubits) @ tensor(
            a_dag(dims), coupling.qubit_indices[1], num_qubits
        )
        c_2 = tensor(a_dag(dims), coupling.qubit_indices[0], num_qubits) @ tensor(
            a(dims), coupling.qubit_indices[1], num_qubits
        )
        coupling_ij = 2 * jnp.pi * coupling.coupling_strength * (c_1 + c_2)

        operators[
            ChannelID(
                qubit_idx=(coupling.qubit_indices[0], coupling.qubit_indices[1]),
                type=TermType.COUPLING,
            ).hash()
        ] = HamiltonianTerm(
            qubit_idx=(coupling.qubit_indices[0], coupling.qubit_indices[1]),
            type=TermType.COUPLING,
            controlable=False,
            operator=coupling_ij,
        )

    return operators


def hamiltonian_fn(
    args: dict[str, SignalParameters],
    t: jnp.ndarray,
    signals: dict[str, Callable[[SignalParameters, jnp.ndarray], jnp.ndarray]],
    hamiltonian_terms: dict[str, HamiltonianTerm],
    static_terms: list[str],
) -> jnp.ndarray:
    # Match the args with signal
    drives = jnp.array([
        signal(args[channel_id], t) * hamiltonian_terms[channel_id].operator
        for channel_id, signal in signals.items()
    ])

    statics = jnp.array([hamiltonian_terms[static_term].operator for static_term in static_terms])

    return jnp.sum(drives, axis=0) + jnp.sum(statics, axis=0)


def pick_non_controlable(channel: str):
    return int(channel[3]) not in [TermType.DRIVE.value, TermType.CONTROL.value]


def pick_drive(channel: str):
    return int(channel[3]) == TermType.DRIVE.value


def pick_static(channel: str):
    return int(channel[3]) == TermType.STATIC.value
