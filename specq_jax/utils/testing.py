import jax
import jax.numpy as jnp
from ..pulse import MultiDragPulse, MultiDragPulseV2, DragPulse
from specq_dev.jax import JaxBasedPulseSequence  # type: ignore
from specq_dev.shared import QubitInformation, Operator  # type: ignore
from ..model import MLPBlackBox
from typing import Callable
from functools import partial
from typing import Union
from specq_dev import shared # type: ignore
from ..simulator import signal_func_v3, simulator as sim, SignalParameters
from .helper import rotating_transmon_hamiltonian

def get_multi_drag_pulse_sequence() -> JaxBasedPulseSequence:

    pulse_length_dt = 80
    sigma_range = [(7, 9), (5, 7), (3, 5), (1, 3)]

    pulse_sequence = JaxBasedPulseSequence(
        pulses=[
            MultiDragPulse(
                total_length=pulse_length_dt,
                num_drag=i,
                min_amp=0,
                max_amp=0.95,
                min_sigma=sigma_range[i - 1][0],
                max_sigma=sigma_range[i - 1][1],
                min_beta=-2,
                max_beta=2,
            )
            for i in range(1, 5)
        ],
        pulse_length_dt=pulse_length_dt,
    )

    return pulse_sequence


def get_multi_drag_pulse_sequence_v2() -> JaxBasedPulseSequence:

    pulse_length_dt = 80
    sigma_range = [(7, 9), (5, 7), (3, 5), (1, 3)]

    pulse_sequence = JaxBasedPulseSequence(
        pulses=[
            MultiDragPulseV2(
                total_length=pulse_length_dt,
                num_drag=i,
                min_amp=0,
                max_amp=1,
                min_sigma=sigma_range[i - 1][0],
                max_sigma=sigma_range[i - 1][1],
                min_beta=-2,
                max_beta=2,
            )
            for i in range(1, 5)
        ],
        pulse_length_dt=pulse_length_dt,
    )

    return pulse_sequence


def get_drag_pulse_sequence(
    qubit_info: QubitInformation,
    amp: float = 0.25,  # NOTE: Choice of amplitude is arbitrary
):
    total_length = 320
    dt = 2 / 9
    area = (
        1 / (2 * qubit_info.drive_strength) / dt
    )  # NOTE: Choice of area is arbitrary e.g. pi pulse
    sigma = (1 * area) / (amp * jnp.sqrt(2 * jnp.pi))

    pulse_sequence = JaxBasedPulseSequence(
        pulses=[
            DragPulse(
                total_length=total_length,
                base_amp=amp,
                base_sigma=float(sigma),
                base_beta=1 / qubit_info.anharmonicity,
                total_amps=1,
                max_amp=0,
                min_amp=0,
                final_amp=1.0,
            )
        ],
        pulse_length_dt=total_length,
    )

    return pulse_sequence


def get_qubit_info() -> QubitInformation:
    return QubitInformation(
        unit="GHz",
        qubit_idx=0,
        anharmonicity=-0.33,
        frequency=5.0,
        drive_strength=0.1,
    )

def linear_transformation(
    params_array: jnp.ndarray,
    u_transformation_matrix: jnp.ndarray,
    d_transformation_matrix: jnp.ndarray,
    u_offset_matrix: jnp.ndarray,
    d_offset_matrix: jnp.ndarray,
) -> jnp.ndarray:
    """Apply linear transformation to the parameters array.

    Args:
        params_array (jnp.ndarray): The array of parameters of the pulse sequence.
        transformation_matrix (jnp.ndarray): The matrix to transform the parameters array.

    Returns:
        jnp.ndarray: The transformed array of parameters to the parameter of operator Wo
    """
    # Expect the Wo params array to be shape of (5,) where [:3] are in the range of [0, 2*pi] and [3:] are in the range of [-1, 1]

    # Apply the linear transformation to the parameters array
    u_param = jnp.dot(u_transformation_matrix, params_array.reshape(-1, 1)).reshape(-1)
    d_param = jnp.dot(d_transformation_matrix, params_array.reshape(-1, 1)).reshape(-1)

    # Apply the offset to the transformed parameters array
    u_param += u_offset_matrix
    d_param += d_offset_matrix

    # Normalize the transformed parameters array
    u_param /= jnp.linalg.norm(u_param)
    d_param /= jnp.linalg.norm(d_param)

    # Apply the constraints to the transformed parameters array
    u_param = 2 * jnp.pi * u_param
    d_param = 2 * (d_param - (1/2)) 

    # NOTE: the parameters that make the operator close to identity is [0, 0, 0, 1, 1]
    # NOTE: if the params_array is all zeros, want to 

    # Concatenate the transformed parameters array
    transformed_params_array = jnp.concatenate((u_param, d_param))

    return transformed_params_array


def generate_random_transformer(
    pulse_sequence: JaxBasedPulseSequence, key: jnp.ndarray, complexity: int
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Generate a random transformer for the pulse sequence.

    Args:
        pulse_sequence (sqj.JaxBasedPulseSequence): The pulse sequence to generate the transformer for.
        key (jnp.ndarray): The random key to generate the transformer.
        complexity (int): The complexity of the transformer.

    Returns:
        Callable[[jnp.ndarray], jnp.ndarray]: The transformer function that takes in the parameters
                                              array and returns the transformed parameters array.
    """

    NUM_U_HERMITIAN_PARAM = 3
    NUM_D_HERMITIAN_PARAM = 2

    sample_key, u_transformer_key, d_transformer_key, u_offset, d_offset = jax.random.split(key, 5)

    # Get the sample parameters
    sample_param = pulse_sequence.sample_params(sample_key)
    # Get the shape of the sample parameters
    param_shape = pulse_sequence.list_of_params_to_array(sample_param).shape
    # Ex: (NUM_HERMITIAN_PARAM, num_pulse_param) @ (num_pulse_param, 1) = (NUM_HERMITIAN_PARAM, 1) -> (NUM_HERMITIAN_PARAM,)
    u_transformer_matrix = jax.random.normal(
        u_transformer_key, (NUM_U_HERMITIAN_PARAM, param_shape[0])
    )
    d_transformer_matrix = jax.random.normal(
        d_transformer_key, (NUM_D_HERMITIAN_PARAM, param_shape[0])
    )

    # Generate the offset for the transformer
    u_offset_matrix = jax.random.uniform(u_offset, (NUM_U_HERMITIAN_PARAM,))
    d_offset_matrix = jax.random.uniform(d_offset, (NUM_D_HERMITIAN_PARAM,))

    partial_linear_transformation = partial(
        linear_transformation,
        u_transformation_matrix=u_transformer_matrix,
        d_transformation_matrix=d_transformer_matrix,
        u_offset_matrix=u_offset_matrix,
        d_offset_matrix=d_offset_matrix,
    )

    return partial_linear_transformation


def ishermitian(matrix: jnp.ndarray) -> bool:
    """Check if the matrix is Hermitian.

    Args:
        matrix (jnp.ndarray): The matrix to check.

    Returns:
        bool: True if the matrix is Hermitian, False otherwise.
    """
    return bool(jnp.allclose(matrix, matrix.conj().T))

def gen_Wo_params_v1(
    key: jnp.ndarray, pulse_sequence: JaxBasedPulseSequence, pulse_parameters
):
    x_transformer_key, y_transformer_key, z_transformer_key = jax.random.split(key, 3)

    random_transformer_x = generate_random_transformer(
        pulse_sequence, x_transformer_key, 0
    )
    random_transformer_y = generate_random_transformer(
        pulse_sequence, y_transformer_key, 0
    )
    random_transformer_z = generate_random_transformer(
        pulse_sequence, z_transformer_key, 0
    )

    x_params = jax.vmap(random_transformer_x)(pulse_parameters)
    y_params = jax.vmap(random_transformer_y)(pulse_parameters)
    z_params = jax.vmap(random_transformer_z)(pulse_parameters)

    Wo_params = {}
    for pauli, params in [("X", x_params), ("Y", y_params), ("Z", z_params)]:
        Wo_params[pauli] = {
            "U": params[:, :3],
            "D": params[:, 3:],
        }

    return Wo_params, (random_transformer_x, random_transformer_y, random_transformer_z)


def gen_Wo_params_v2(
    key: jnp.ndarray, pulse_sequence: JaxBasedPulseSequence, pulse_parameters
):
    transformer = MLPBlackBox(
        feature_size=5, hidden_sizes_1=[5], hidden_sizes_2=[5]
    )
    init_param = transformer.init(
        key, jnp.ones((1, pulse_parameters.shape[1])), training=False
    )

    Wo_params = transformer.apply(
        init_param,
        pulse_parameters,
        training=False,
    )

    noise_transformer = lambda x: transformer.apply(init_param, x, training=False)

    return Wo_params, noise_transformer



def gen_mock_data(
    key: jnp.ndarray,
    batch_size: int = 1500, 
    qubit_info: Union[QubitInformation, None] = None, 
    pulse_sequence: Union[JaxBasedPulseSequence, None] = None,
    hamiltonian: Union[Callable, None] = None,
    hamiltonian_transformer: list[Callable] = [],
):

    qubit_info = get_qubit_info() if qubit_info is None else qubit_info
    pulse_sequence = get_multi_drag_pulse_sequence_v2() if pulse_sequence is None else pulse_sequence
    dt = 2 / 9

    config = shared.ExperimentConfiguration(
        qubits=[qubit_info],
        expectation_values_order=shared.default_expectation_values,
        parameter_names=pulse_sequence.get_parameter_names(),
        backend_name="ideal_simulator",
        shots=1,
        EXPERIMENT_IDENTIFIER="test",
        EXPERIMENT_TAGS=["test"],
        description="generate mock data for testing",
        device_cycle_time_ns=dt,
        sequence_duration_dt=pulse_sequence.pulse_length_dt,
        instance="test",
        sample_size=batch_size,
    )

    t_eval = jnp.linspace(0, pulse_sequence.pulse_length_dt * dt, 100)

    signal = signal_func_v3(
        pulse_sequence.get_envelope, drive_frequency=qubit_info.frequency, dt=dt
    )
    _hamiltonian = partial(
        rotating_transmon_hamiltonian, qubit_info=qubit_info, signal=signal
    ) if hamiltonian is None else hamiltonian

    original_hamiltonian = _hamiltonian

    # Transform the Hamiltonian
    for transformer in hamiltonian_transformer:
        _hamiltonian = transformer(_hamiltonian)

    simulator = jax.jit(
        partial(
            sim,
            t_eval=t_eval,
            hamiltonian=_hamiltonian,
            y0=jnp.eye(2, dtype=jnp.complex64),
            t0=0.0,
            t1=pulse_sequence.pulse_length_dt * dt,
        )
    )

    _pulse_parameters = []
    signal_params = []
    for i in range(batch_size):
        key, subkey = jax.random.split(key)
        params = pulse_sequence.sample_params(subkey)
        signal_params.append(SignalParameters(pulse_params=params, phase=0))
        _pulse_parameters.append(pulse_sequence.list_of_params_to_array(params))

    pulse_parameters = jnp.array(_pulse_parameters)

    _unitaries: list[jnp.ndarray] = []
    for params in signal_params:
        _unitaries.append(simulator(params))

    unitaries = jnp.array(_unitaries)

    return config, pulse_sequence, pulse_parameters, unitaries, original_hamiltonian

def get_PSD_v1():

    alpha = 1
    beta = 0.8
    peak = 15

    def spectrum(freqency):
        return (1 / (freqency + 1) ** (alpha)) + (
            beta * jnp.exp(-((freqency - peak) ** 2) / 10)
        )
    
    return spectrum

def noise_from_spectrum(key: jnp.ndarray, spectrum: jnp.ndarray, frequency_space: jnp.ndarray):
    
    phases = jax.random.uniform(key, shape=spectrum.shape) * 2 * jnp.pi
    dw = frequency_space[1] - frequency_space[0]

    a = lambda t: 2 * jnp.sum(jnp.sqrt(spectrum * dw) * jnp.cos(frequency_space * t + phases))
    return a