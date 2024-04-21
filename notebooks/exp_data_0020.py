import specq_dev.specq.shared as specq  # type: ignore
from specq_jax.pulse import MultiDragPulse  # type: ignore
from specq_dev.specq.jax import JaxBasedPulseSequence  # type: ignore
import jax.numpy as jnp
import jax
import numpy as np
from specq_jax.core import get_simulator, rotating_duffling_oscillator_hamiltonian


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


def get_exp_data(hamiltonian=rotating_duffling_oscillator_hamiltonian):
    # Load the data from the experiment
    exp_data = specq.ExperimentDataV3.from_folder(
        "../../specq-experiment/datasets/0020"
    )

    print(f"Loaded data from {exp_data.experiment_config.EXPERIMENT_IDENTIFIER}")

    # Setup the simulator
    dt = exp_data.experiment_config.device_cycle_time_ns
    qubit_info = exp_data.experiment_config.qubits[0]
    pulse_sequence = get_multi_drag_pulse_sequence()
    t_eval = jnp.linspace(
        0, pulse_sequence.pulse_length_dt * dt, pulse_sequence.pulse_length_dt
    )
    simulator = get_simulator(
        qubit_info=qubit_info, t_eval=t_eval, hamiltonian=hamiltonian
    )

    # Get the waveforms for each pulse parameters to get the unitaries
    waveforms = []
    for pulse_params in exp_data.get_parameters_dict_list():
        waveforms.append(pulse_sequence.get_waveform(pulse_params))

    waveforms = jnp.array(waveforms)

    print(
        f"Prepared the waveforms for the experiment {exp_data.experiment_config.EXPERIMENT_IDENTIFIER}"
    )

    # jit the simulator
    jitted_simulator = jax.jit(simulator)
    # batch the simulator
    batched_simulator = jax.vmap(jitted_simulator, in_axes=(0))
    # Get the unitaries
    unitaries = batched_simulator(waveforms)

    print(
        f"Got the unitaries for the experiment {exp_data.experiment_config.EXPERIMENT_IDENTIFIER}"
    )

    # Get the final unitaries
    unitaries = np.array(unitaries[:, -1, :, :])
    # Get the expectation values from the experiment
    expectations = exp_data.get_expectation_values()
    # Get the pulse parameters
    pulse_parameters = exp_data.parameters

    print(
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
