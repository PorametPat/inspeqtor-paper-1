import jax
import jax.numpy as jnp
from dataclasses import dataclass
import typing
from ..data import (
    ExpectationValue,
    QubitInformation,
    ExperimentConfiguration,
    ExperimentData,
)
from ..pulse import JaxBasedPulse, JaxBasedPulseSequence
from ..typing import ParametersDictType
from ..physics import SignalParameters
from ..constant import X, Y, Z, default_expectation_values_order
from ..decorator import not_yet_tested

def center_location(num_of_pulse_in_dd: int, total_time_dt: int | float):
    center_locations = (
        jnp.array(
            [(k - 0.5) / num_of_pulse_in_dd for k in range(1, num_of_pulse_in_dd + 1)]
        )
        * total_time_dt
    )  # ideal CPMG pulse locations for x-axis
    return center_locations


def drag_envelope(
    amp: float, sigma: float, beta: float, center: float, final_amp: float | None = None
):

    # beta is the coefficient of the derivative term
    # beta = 0 is the standard gaussian pulse
    # beta = lambda / alpha, where alpha is the anhamonicity setting lambda = 1 is to
    # cancel out the leakage error

    g = lambda t: amp * jnp.exp(-((t - center) ** 2) / (2 * sigma**2))
    f_prime = lambda t: g(t) + 1j * beta * (-1 * (t - center) / sigma**2) * g(t)

    if final_amp is None:
        final_amp = amp

    # return f_prime
    return lambda t: final_amp * (f_prime(t) - f_prime(-1)) / (1 - f_prime(-1))


@dataclass
class MultiDragPulseV2(JaxBasedPulse):
    total_length: int

    num_drag: int = 1
    min_amp: float = 0
    max_amp: float = 1
    min_sigma: float = 0.1
    max_sigma: float = 10
    min_beta: float = 0.1
    max_beta: float = 10

    def __post_init__(self):
        self.t_eval = jnp.arange(self.total_length)

    def sample_params(self, key: jnp.ndarray) -> ParametersDictType:

        params: ParametersDictType = dict()
        for i in range(self.num_drag):

            # Split the key
            key, amp_key, sigma_key, beta_key = jax.random.split(key, num=4)
            random_amps = jax.random.uniform(
                amp_key, shape=(1,), minval=self.min_amp, maxval=self.max_amp
            )

            random_sigmas = jax.random.uniform(
                sigma_key, shape=(1,), minval=self.min_sigma, maxval=self.max_sigma
            )

            random_betas = jax.random.uniform(
                beta_key, shape=(1,), minval=self.min_beta, maxval=self.max_beta
            )

            param = {
                f"{i}/amp": float(random_amps[0]),
                f"{i}/sigma": float(random_sigmas[0]),
                f"{i}/beta": float(random_betas[0]),
            }
            params.update(param)

        return params

    def get_bounds(self) -> tuple[ParametersDictType, ParametersDictType]:

        lower = {}
        upper = {}

        for i in range(self.num_drag):
            lower[f"{i}/amp"] = self.min_amp
            lower[f"{i}/sigma"] = self.min_sigma
            lower[f"{i}/beta"] = self.min_beta

            upper[f"{i}/amp"] = self.max_amp
            upper[f"{i}/sigma"] = self.max_sigma
            upper[f"{i}/beta"] = self.max_beta

        return lower, upper

    def get_waveform_manual(self, params: ParametersDictType) -> jnp.ndarray:

        # The center location of each drag pulse
        center_locations = center_location(self.num_drag, self.total_length)

        # Initialize the base pulse
        base_pulse = jnp.zeros(self.total_length, dtype=jnp.complex64)

        # For each drag pulse, add the drag envelope
        for idx, center in enumerate(center_locations):
            base_pulse += drag_envelope(
                params[f"{idx}/amp"],
                params[f"{idx}/sigma"],
                params[f"{idx}/beta"],
                center,
            )(self.t_eval)

        # Normalize the pulse by 1 / (2 ** num_drag ) to avoid the pulse amplitude to be too large
        base_pulse /= 2**self.num_drag

        return base_pulse

    def get_envelope(
        self, params: dict[str, float]
    ) -> typing.Callable[..., typing.Any]:

        # The center location of each drag pulse
        center_locations = center_location(self.num_drag, self.total_length)

        # Callables
        envelopes = []
        for idx, center in enumerate(center_locations):

            envelopes.append(
                drag_envelope(
                    params[f"{idx}/amp"],
                    params[f"{idx}/sigma"],
                    params[f"{idx}/beta"],
                    center,
                )
            )

        # Return the sum of the envelopes
        return lambda t: sum([envelope(t) for envelope in envelopes]) / (
            2**self.num_drag
        )

    def get_waveform(self, params: ParametersDictType) -> jnp.ndarray:
        return self.get_envelope(params)(self.t_eval)


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


def get_mock_qubit_information() -> QubitInformation:
    return QubitInformation(
        unit="GHz",
        qubit_idx=0,
        anharmonicity=-0.2,
        frequency=5.0,
        drive_strength=0.1,
    )


def get_mock_prefined_exp_v1(sample_size: int = 10, shots: int = 1000):

    qubit_info = get_mock_qubit_information()
    pulse_sequence = get_multi_drag_pulse_sequence_v2()

    config = ExperimentConfiguration(
        qubits=[qubit_info],
        expectation_values_order=default_expectation_values_order,
        parameter_names=pulse_sequence.get_parameter_names(),
        backend_name="fake_ibm_test",
        shots=shots,
        EXPERIMENT_IDENTIFIER="test",
        EXPERIMENT_TAGS=["test"],
        description="Generated for test",
        device_cycle_time_ns=2 / 9,
        sequence_duration_dt=pulse_sequence.pulse_length_dt,
        instance="inspeqtor/tester",
        sample_size=sample_size,
    )

    return qubit_info, pulse_sequence, config


@dataclass
class JaxDragPulse(JaxBasedPulse):
    total_length: int
    base_amp: float
    base_sigma: float
    base_beta: float
    total_amps: int

    min_amp: float = -1
    max_amp: float = 1

    def __post_init__(self):
        self.t_eval = jnp.arange(self.total_length)
        self.base_drag = drag_envelope(
            amp=self.base_amp,
            sigma=self.base_sigma,
            beta=self.base_beta,
            center=self.total_length // 2,
        )(self.t_eval)

    def sample_params(self, key: jax.Array) -> ParametersDictType:

        params: ParametersDictType = dict()

        shifted_amps = jax.random.uniform(
            key, (self.total_amps, 2), jnp.float32, self.min_amp, self.max_amp
        )

        for i in range(self.total_amps):
            param = {
                f"amp/real/{i}": float(shifted_amps[i, 0]),
                f"amp/imag/{i}": float(shifted_amps[i, 1]),
            }
            params.update(param)

        return params

    def get_bounds(self) -> tuple[dict[str, float], dict[str, float]]:

        lower = {}
        upper = {}

        for i in range(self.total_amps):
            lower[f"amp/real/{i}"] = self.min_amp
            upper[f"amp/real/{i}"] = self.max_amp
            lower[f"amp/imag/{i}"] = self.min_amp
            upper[f"amp/imag/{i}"] = self.max_amp

        return lower, upper

    def get_waveform(self, params: ParametersDictType) -> jnp.ndarray:

        # Turn params to array with real and imaginary parts
        amps = jnp.array(
            [
                params[f"amp/real/{i}"] + 1j * params[f"amp/imag/{i}"]
                for i in range(self.total_amps)
            ]
        )
        # Zero pad to total length
        amps = jnp.pad(amps, (0, self.total_length - self.total_amps))
        # Add base drag
        amps = amps + self.base_drag

        return amps


def get_jax_based_pulse_sequence() -> JaxBasedPulseSequence:

    drive_str = 0.11
    dt = 2 / 9
    total_length = 80
    amp = 0.8
    area = 1 / (2 * drive_str * amp) / dt
    sigma = (1 * area) / (amp * jnp.sqrt(2 * jnp.pi))

    pulse_sequence = JaxBasedPulseSequence(
        pulses=[
            JaxDragPulse(
                total_length=total_length,
                base_amp=amp,
                base_sigma=float(sigma),
                base_beta=-1,
                total_amps=80,
                min_amp=-0.1,
                max_amp=0.1,
            )
        ],
        pulse_length_dt=total_length,
    )

    return pulse_sequence


@dataclass
class DragPulse(JaxBasedPulse):
    total_length: int
    base_amp: float
    base_sigma: float
    base_beta: float
    total_amps: int

    min_amp: float = -1
    max_amp: float = 1
    final_amp: float | None = None

    def __post_init__(self):
        self.t_eval = jnp.arange(self.total_length)
        self.base_drag = drag_envelope(
            amp=self.base_amp,
            sigma=self.base_sigma,
            beta=self.base_beta,
            center=self.total_length // 2,
            final_amp=self.final_amp,
        )(self.t_eval)

    def sample_params(self, key: jnp.ndarray) -> ParametersDictType:

        params: ParametersDictType = dict()
        for i in range(self.total_amps):

            # Split the key
            key, subkey = jax.random.split(key)
            random_amps = jax.random.uniform(
                subkey, shape=(2,), minval=self.min_amp, maxval=self.max_amp
            )

            param = {
                f"amp/real/{i}": float(random_amps[0]),
                f"amp/imag/{i}": float(random_amps[1]),
            }
            params.update(param)

        return params

    def get_bounds(self) -> tuple[ParametersDictType, ParametersDictType]:

        lower = {}
        upper = {}

        for i in range(self.total_amps):
            lower[f"amp/real/{i}"] = self.min_amp
            lower[f"amp/imag/{i}"] = self.min_amp

            upper[f"amp/real/{i}"] = self.max_amp
            upper[f"amp/imag/{i}"] = self.max_amp

        return lower, upper

    def get_waveform(self, params: ParametersDictType) -> jnp.ndarray:
        return self.get_envelope(params)(self.t_eval)

    def get_envelope(
        self, params: ParametersDictType
    ) -> typing.Callable[..., typing.Any]:

        return drag_envelope(
            amp=self.base_amp,
            sigma=self.base_sigma,
            beta=self.base_beta,
            center=self.total_length // 2,
            final_amp=self.final_amp,
        )


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


def rotating_transmon_hamiltonian(
    params: SignalParameters,
    t: jnp.ndarray,
    qubit_info: QubitInformation,
    signal: typing.Callable[..., jnp.ndarray],
) -> jnp.ndarray:
    """Rotating frame hamiltonian of the transmon model

    Args:
        params (HamiltonianParameters): The parameter of the pulse for hamiltonian
        t (jnp.ndarray): The time to evaluate the Hamiltonian
        qubit_info (QubitInformation): The information of qubit
        signal (Callable[..., jnp.ndarray]): The pulse signal

    Returns:
        jnp.ndarray: The Hamiltonian
    """
    a0 = 2 * jnp.pi * qubit_info.frequency
    a1 = 2 * jnp.pi * qubit_info.drive_strength

    f3 = lambda _params, t: a1 * signal(_params, t)
    f_sigma_x = lambda _params, t: f3(_params, t) * jnp.cos(a0 * t)
    f_sigma_y = lambda _params, t: f3(_params, t) * jnp.sin(a0 * t)

    return f_sigma_x(params, t) * X - f_sigma_y(params, t) * Y


def transmon_hamiltonian(
    params: SignalParameters,
    t: jnp.ndarray,
    qubit_info: QubitInformation,
    signal: typing.Callable[..., jnp.ndarray],
) -> jnp.ndarray:
    """Lab frame hamiltonian of the transmon model

    Args:
        params (HamiltonianParameters): The parameter of the pulse for hamiltonian
        t (jnp.ndarray): The time to evaluate the Hamiltonian
        qubit_info (QubitInformation): The information of qubit
        signal (Callable[..., jnp.ndarray]): The pulse signal

    Returns:
        jnp.ndarray: The Hamiltonian
    """

    a0 = 2 * jnp.pi * qubit_info.frequency
    a1 = 2 * jnp.pi * qubit_info.drive_strength

    return ((a0 / 2) * Z) + (a1 * signal(params, t) * X)

@not_yet_tested
def detune_hamiltonian(
    hamiltonian: typing.Callable[..., jnp.ndarray],
    detune: float,
):
    
    def detuned_hamiltonian(params: SignalParameters, t: jnp.ndarray) -> jnp.ndarray:
        return hamiltonian(params, t) + detune * Z

    return detuned_hamiltonian


@not_yet_tested
def get_PSD_v1():

    alpha = 1
    beta = 0.8
    peak = 15

    def spectrum(freqency):
        return (1 / (freqency + 1) ** (alpha)) + (
            beta * jnp.exp(-((freqency - peak) ** 2) / 10)
        )
    
    return spectrum


@not_yet_tested
def noise_from_spectrum(key: jnp.ndarray, spectrum: jnp.ndarray, frequency_space: jnp.ndarray):
    
    phases = jax.random.uniform(key, shape=spectrum.shape) * 2 * jnp.pi
    dw = frequency_space[1] - frequency_space[0]

    a = lambda t: 2 * jnp.sum(jnp.sqrt(spectrum * dw) * jnp.cos(frequency_space * t + phases))
    return a


@not_yet_tested
def signal_with_colored_noise(
        signal: typing.Callable[[SignalParameters, jnp.ndarray], jnp.ndarray],
        noise_func: typing.Callable[[jnp.ndarray], typing.Callable[[float], float]],
        spectrum: typing.Callable[[jnp.ndarray], jnp.ndarray],
        frequency_space: jnp.ndarray,
    ):
    """Make the envelope function into signal with drive frequency

    Args:
        get_envelope (Callable): The envelope function in unit of dt
        drive_frequency (float): drive freuqency in unit of GHz
        dt (float): The dt provived will be used to convert envelope unit to ns,
                    set to 1 if the envelope function is already in unit of ns
    """

    # def signal(pulse_parameters: SignalParameters, t: jnp.ndarray):
    #     return jnp.real(
    #         get_envelope(pulse_parameters.pulse_params)(t / dt)
    #         * jnp.exp(
    #             1j * ((2 * jnp.pi * drive_frequency * t) + pulse_parameters.phase)
    #         )
    #     ) + 0.01 * noise_func(pulse_parameters.noise_key, spectrum(frequency_space), frequency_space)(t)

    # return signal

    
    def noisy_signal(pulse_parameters: SignalParameters, t: jnp.ndarray):
        return signal(pulse_parameters, t) + 0.01 * noise_func(pulse_parameters.noise_key, spectrum(frequency_space), frequency_space)(t)

    return noisy_signal