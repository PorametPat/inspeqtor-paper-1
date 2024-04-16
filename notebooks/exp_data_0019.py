import specq_dev.specq.shared as specq # type: ignore
import jax.numpy as jnp
import jax
import numpy as np
from specq_jax.core import get_simulator
from specq_jax.pulse import DragPulse, JaxBasedPulseSequence # type: ignore

def get_drag_pulse_sequence(
    dt: float,
    drive_str: float,
) -> JaxBasedPulseSequence:

    total_length = 80
    amp = 0.8
    area = 1 / (2 * drive_str * amp) / dt
    sigma = (1 * area) / (amp * jnp.sqrt(2 * jnp.pi))

    pulse_sequence = JaxBasedPulseSequence(
        pulses=[
            DragPulse(
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

def get_exp_data():
    # Load the data from the experiment
    exp_data = specq.ExperimentDataV3.from_folder("../specq-experiment/datasets/0019")

    print(f"Loaded data from {exp_data.experiment_config.EXPERIMENT_IDENTIFIER}")

    # Setup the simulator
    dt = exp_data.experiment_config.device_cycle_time_ns
    qubit_info = exp_data.experiment_config.qubits[0]
    pulse_sequence = get_drag_pulse_sequence(dt=dt, drive_str=qubit_info.drive_strength)
    t_eval = jnp.linspace(
        0, pulse_sequence.pulse_length_dt * dt, pulse_sequence.pulse_length_dt
    )
    simulator = get_simulator(qubit_info=qubit_info, t_eval=t_eval)

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

    return pulse_parameters, unitaries, expectations, pulse_sequence, simulator
