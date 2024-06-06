import jax
import jax.numpy as jnp
from ..pulse import MultiDragPulse, MultiDragPulseV2, DragPulse
from specq_dev.jax import JaxBasedPulseSequence  # type: ignore
from specq_dev.shared import QubitInformation, Operator  # type: ignore

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

