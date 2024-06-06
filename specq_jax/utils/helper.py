import jax.numpy as jnp
from ..simulator import SignalParameters
from typing import Callable
from specq_dev.shared import QubitInformation # type: ignore
from ..core import X, Y, Z


def check_diagonal_matrix(matrix):
    return jnp.count_nonzero(matrix - jnp.diag(jnp.diagonal(matrix))) == 0


def rotating_transmon_hamiltonian(
    params: SignalParameters,
    t: jnp.ndarray,
    qubit_info: QubitInformation,
    signal: Callable[..., jnp.ndarray],
)-> jnp.ndarray:
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
    signal: Callable[..., jnp.ndarray],
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
