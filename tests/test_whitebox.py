import pytest
from specq_dev.jax import JaxBasedPulseSequence # type: ignore
from specq_dev.shared import QubitInformationV3 # type: ignore
from specq_jax.pulse import MultiDragPulse
from specq_jax.core import get_simulator
import jax
import jax.numpy as jnp

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

def test_run():

    batch_size = 10
    key = jax.random.PRNGKey(0)
    qubit_info = QubitInformationV3(
        unit="GHz",
        qubit_idx=0,
        anharmonicity=-0.33,
        frequency=5.0,
        drive_strength=0.1,
    )
    dt = 2 / 9

    # Get the pulse sequence
    pulse_sequence = get_multi_drag_pulse_sequence()
    t_eval = t_eval = jnp.linspace(
        0, pulse_sequence.pulse_length_dt * dt, pulse_sequence.pulse_length_dt
    )

    # Sampling the pulse parameters
    # Get the waveforms for each pulse parameters to get the unitaries
    waveforms = []
    for i in range(batch_size):
        key, subkey = jax.random.split(key)
        pulse_params = pulse_sequence.sample_params(subkey)
        waveforms.append(pulse_sequence.get_waveform(pulse_params))

    waveforms = jnp.array(waveforms)

    # Get the simualtor
    simulator = get_simulator(qubit_info=qubit_info, t_eval=t_eval)

    # Solve for the unitaries
    # jit the simulator
    jitted_simulator = jax.jit(simulator)
    # batch the simulator
    batched_simulator = jax.vmap(jitted_simulator, in_axes=(0))
    # Get the unitaries
    unitaries = batched_simulator(waveforms)
    # Assert the unitaries shape
    assert unitaries.shape == (batch_size, pulse_sequence.pulse_length_dt, 2, 2)
