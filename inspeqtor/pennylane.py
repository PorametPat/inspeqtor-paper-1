import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pennylane as qml  # type: ignore
from typing import Callable
from .data import QubitInformation

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
    qubit_info: QubitInformation,
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
    qubit_info: QubitInformation, signal: Callable
) -> qml.pulse.parametrized_hamiltonian.ParametrizedHamiltonian:
    a0 = 2 * jnp.pi * qubit_info.frequency
    a1 = 2 * jnp.pi * qubit_info.drive_strength

    f3 = lambda params, t: a1 * signal(params, t)  # * ((a0 * t / 2)**2)
    f_sigma_x = lambda params, t: f3(params, t) * jnp.cos(a0 * t)
    f_sigma_y = lambda params, t: f3(params, t) * jnp.sin(a0 * t)

    return f_sigma_x * qml.PauliX(0) + f_sigma_y * qml.PauliY(0)


def transmon_hamiltonian(
    qubit_info: QubitInformation, waveform: Callable
) -> qml.pulse.parametrized_hamiltonian.ParametrizedHamiltonian:
    H_0 = qml.pulse.transmon_interaction(
        qubit_freq=[qubit_info.frequency], connections=[], coupling=[], wires=[0]
    )
    H_d0 = qml.pulse.transmon_drive(
        waveform, 0, qubit_info.frequency, 0
    )  # NOTE: f2 must be real function.

    return H_0 + H_d0


def rotating_transmon_hamiltonian(
    qubit_info: QubitInformation, signal: Callable
):

    a0 = 2 * jnp.pi * qubit_info.frequency
    a1 = 2 * jnp.pi * qubit_info.drive_strength

    f3 = lambda params, t: a1 * signal(params, t)
    f_sigma_x = lambda params, t: f3(params, t) * jnp.cos(a0 * t)
    f_sigma_y = lambda params, t: -1 * f3(params, t) * jnp.sin(a0 * t)

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
    qubit_info: QubitInformation,
    t_eval: jnp.ndarray,
    hamiltonian: Callable[
        [QubitInformation, Callable],
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
