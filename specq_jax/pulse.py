import jax.numpy as jnp
import jax
from dataclasses import dataclass
from specq_dev.jax import JaxBasedPulse  # type: ignore
import specq_dev.shared as specq  # type: ignore
from specq_dev.shared import ParametersDictType  # type: ignore
from typing import Any, Callable


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

    def sample_params(self, key: jnp.ndarray) -> specq.ParametersDictType:

        params: specq.ParametersDictType = dict()
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

    def get_bounds(self) -> tuple[specq.ParametersDictType, specq.ParametersDictType]:

        lower = {}
        upper = {}

        for i in range(self.total_amps):
            lower[f"amp/real/{i}"] = self.min_amp
            lower[f"amp/imag/{i}"] = self.min_amp

            upper[f"amp/real/{i}"] = self.max_amp
            upper[f"amp/imag/{i}"] = self.max_amp

        return lower, upper

    def get_waveform(self, params: specq.ParametersDictType) -> jnp.ndarray:

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

    def get_envelope(
        self, params: ParametersDictType
    ) -> Callable[..., Any]:

        return drag_envelope(
            amp=self.base_amp,
            sigma=self.base_sigma,
            beta=self.base_beta,
            center=self.total_length // 2,
            final_amp=self.final_amp,
        )


@dataclass
class MultiDragPulse(JaxBasedPulse):
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

    def sample_params(self, key: jnp.ndarray) -> specq.ParametersDictType:

        params: specq.ParametersDictType = dict()
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

    def get_bounds(self) -> tuple[specq.ParametersDictType, specq.ParametersDictType]:

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

    def get_waveform(self, params: specq.ParametersDictType) -> jnp.ndarray:

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

        # Normalize the pulse by 1 / (self.num_drag ** 2 ) to avoid the pulse amplitude to be too large
        base_pulse /= self.num_drag**2

        return base_pulse


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

    def sample_params(self, key: jnp.ndarray) -> specq.ParametersDictType:

        params: specq.ParametersDictType = dict()
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

    def get_bounds(self) -> tuple[specq.ParametersDictType, specq.ParametersDictType]:

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

    def get_waveform(self, params: specq.ParametersDictType) -> jnp.ndarray:

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

    def get_envelope(self, params: dict[str, float]) -> Callable[..., Any]:

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
