from flax import struct
import jax
import jax.numpy as jnp
import typing
from dataclasses import dataclass
from enum import Enum
import diffrax  # type: ignore
from .typing import ParametersDictType
from .qiskit import IBMQDeviceProperties
from .data import QubitInformation, ExpectationValue
from .constant import X, Y, Z

# Forest-Benchmarking
from forest.benchmarking.observable_estimation import ExperimentResult, ExperimentSetting, TensorProductState, PauliTerm  # type: ignore
from forest.benchmarking.tomography import linear_inv_process_estimate, pgdb_process_estimate  # type: ignore
from forest.benchmarking.operator_tools.superoperator_transformations import choi2superop  # type: ignore


@struct.dataclass
class SignalParameters:
    pulse_params: list[ParametersDictType]
    phase: float
    random_key: None | jnp.ndarray = None


class SimulatorFnV2(typing.Protocol):
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


HamiltonianArgs = typing.TypeVar("HamiltonianArgs")


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


def normalizer(matrix: jnp.ndarray) -> jnp.ndarray:
    """Normalize the given matrix with QR decomposition and return matrix Q
       which is unitary

    Args:
        matrix (jnp.ndarray): The matrix to normalize to unitary matrix

    Returns:
        jnp.ndarray: The unitary matrix
    """
    return jnp.linalg.qr(matrix).Q  # type: ignore


def solver(
    args: HamiltonianArgs,
    t_eval: jnp.ndarray,
    hamiltonian: typing.Callable,
    y0: jnp.ndarray,
    t0: float,
    t1: float,
    rtol: float = 1e-7,
    atol: float = 1e-7,
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
        t0=t0,
        t1=t1,
        dt0=None,
        stepsize_controller=diffrax.PIDController(
            rtol=rtol,
            atol=atol,
        ),
        y0=y0,
        args=args,
        saveat=diffrax.SaveAt(ts=t_eval),
        max_steps=max_steps,
    )

    # Normailized the solution
    return jax.vmap(normalizer)(solution.ys)


def auto_rotating_frame_hamiltonian(
    hamiltonian: typing.Callable[[HamiltonianArgs, jnp.ndarray], jnp.ndarray],
    frame: jnp.ndarray,
    explicit_deriv: bool = False,
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


def get_coupling_strengths(
    backend_properties: IBMQDeviceProperties,
) -> list[CouplingInformation]:
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
    coupling_constants: list[CouplingInformation] = []
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
    qubit_informations: list[QubitInformation],
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
    signals: dict[str, typing.Callable[[SignalParameters, jnp.ndarray], jnp.ndarray]],
    hamiltonian_terms: dict[str, HamiltonianTerm],
    static_terms: list[str],
) -> jnp.ndarray:
    # Match the args with signal
    drives = jnp.array(
        [
            signal(args[channel_id], t) * hamiltonian_terms[channel_id].operator
            for channel_id, signal in signals.items()
        ]
    )

    statics = jnp.array(
        [hamiltonian_terms[static_term].operator for static_term in static_terms]
    )

    return jnp.sum(drives, axis=0) + jnp.sum(statics, axis=0)


def pick_non_controlable(channel: str):
    return int(channel[3]) not in [TermType.DRIVE.value, TermType.CONTROL.value]


def pick_drive(channel: str):
    return int(channel[3]) == TermType.DRIVE.value


def pick_static(channel: str):
    return int(channel[3]) == TermType.STATIC.value


def signal_func_v3(get_envelope: typing.Callable, drive_frequency: float, dt: float):
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

def signal_func_v5(get_envelope: typing.Callable, drive_frequency: float, dt: float):
    """Make the envelope function into signal with drive frequency

    Args:
        get_envelope (Callable): The envelope function in unit of dt
        drive_frequency (float): drive freuqency in unit of GHz
        dt (float): The dt provived will be used to convert envelope unit to ns,
                    set to 1 if the envelope function is already in unit of ns
    """

    def signal(pulse_parameters: list[dict[str, float]], t: jnp.ndarray):
        return jnp.real(
            get_envelope(pulse_parameters)(t / dt)
            * jnp.exp(
                1j * (2 * jnp.pi * drive_frequency * t)
            )
        )

    return signal




def gate_fidelity(U: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:

    up = jnp.trace(U.conj().T @ V)
    down = jnp.sqrt(jnp.trace(U.conj().T @ U) * jnp.trace(V.conj().T @ V))

    return jnp.abs(up / down) ** 2


def process_fidelity(superop: jnp.ndarray, target_superop: jnp.ndarray) -> jnp.ndarray:
    # Calculate the fidelity
    # TODO: check if this is correct
    fidelity = jnp.trace(jnp.matmul(target_superop.conj().T, superop)) / 4
    return fidelity


def avg_gate_fidelity_from_superop(
    U: jnp.ndarray, target_U: jnp.ndarray
) -> jnp.ndarray:
    dim = 2
    avg_fid = (dim * process_fidelity(U, target_U) + 1) / (dim + 1)
    return jnp.real(avg_fid)


def to_superop(U: jnp.ndarray) -> jnp.ndarray:
    return jnp.kron(U.conj(), U)


def state_tomography(exp_X: jnp.ndarray, exp_Y: jnp.ndarray, exp_Z: jnp.ndarray):
    return (jnp.eye(2) + exp_X * X + exp_Y * Y + exp_Z * Z) / 2


def check_valid_density_matrix(rho: jnp.ndarray):
    # Check if the density matrix is valid
    assert jnp.allclose(jnp.trace(rho), 1.0), "Density matrix is not trace 1"
    assert jnp.allclose(rho, rho.conj().T), "Density matrix is not Hermitian"


def check_hermitian(op: jnp.ndarray):
    assert jnp.allclose(op, op.conj().T), "Matrix is not Hermitian"


def process_tomography(
    rho_0: jnp.ndarray,
    rho_1: jnp.ndarray,
    rho_p: jnp.ndarray,
    rho_m: jnp.ndarray,
):

    Lambda = (1 / 2) * jnp.block([[jnp.eye(2), X], [X, -jnp.eye(2)]])

    rho_p1 = rho_0
    rho_p4 = rho_1
    rho_p2 = rho_p - 1j * rho_m - (1 - 1j) * (rho_p1 + rho_p4) / 2
    rho_p3 = rho_p + 1j * rho_m - (1 + 1j) * (rho_p1 + rho_p4) / 2

    # NOTE: This is difference from the block 8.5 in Nielsen and Chuang.
    Rho = jnp.block([[rho_p1, rho_p3], [rho_p2, rho_p4]])
    # Rho = jnp.block([[rho_p1, rho_p2], [rho_p3, rho_p4]])

    # return choi2superop(Rho)

    Chi = Lambda @ Rho @ Lambda

    # Check hermitain
    check_hermitian(Chi)

    # From process matrix to choi matrix
    P_0 = jnp.eye(2)
    P_1 = X
    # P_2 = -1*Y
    P_2 = Y
    P_3 = Z

    Ps = [P_0, P_1, P_2, P_3]
    choi_matrix = jnp.zeros((4, 4))
    for m, pauli_m in enumerate(Ps):
        for n, pauli_n in enumerate(Ps):
            superket_m = pauli_m.flatten().reshape(-1, 1)
            superbra_n = pauli_n.flatten().conj()
            # Outer product
            choi_basis = Chi[m, n] * jnp.outer(superket_m, superbra_n)
            choi_matrix += choi_basis

    return choi2superop(choi_matrix)


initial_state_mapping = {
    "0": "Z+",
    "1": "Z-",
    "+": "X+",
    "-": "X-",
    "r": "Y+",
    "l": "Y-",
}


def forest_process_tomography(
    expvals: list[ExpectationValue],
    is_linear_inv: bool = True,
):
    """Calculate the process tomography estimate using the Forest-benchmarking library.

    Args:
        expvals (list[sq.data.ExpectationValue]): The list of expectation values.

    Returns:
        _type_: The process tomography estimate in the Choi matrix representation.
    """
    # Extract results
    exp_results = []

    for expval in expvals:

        exp_results.append(
            ExperimentResult(
                setting=ExperimentSetting(
                    observable=PauliTerm.from_list(
                        [(expval.observable, 0)]
                    ),  # NOTE: Assumes qubit 0
                    in_state=TensorProductState.from_str(
                        f"{initial_state_mapping[expval.initial_state]}_0"  # NOTE: Assumes qubit 0
                    ),
                ),
                expectation=expval.expectation_value,
                total_counts=1000,  # TODO: Should be set to the actual number of counts?
            )
        )

    if is_linear_inv:
        est = linear_inv_process_estimate(exp_results, [0])  # NOTE: Assumes qubit 0
    else:
        est = pgdb_process_estimate(exp_results, [0])  # NOTE: Assumes qubit 0

    return est
