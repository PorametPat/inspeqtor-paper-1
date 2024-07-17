import jax
from jax import numpy as jnp
import inspeqtor as isq


def test_jax_DRAG_pulse():

    pulse_sequence = isq.utils.predefined.get_multi_drag_pulse_sequence_v2()

    key = jax.random.PRNGKey(0)
    params = pulse_sequence.sample_params(key)
    waveform = pulse_sequence.get_waveform(params)

    # Check that the waveform is of the correct length
    assert waveform.shape == (pulse_sequence.pulse_length_dt,)

    # Check to_dict and from_dict
    pulse_sequence_dict = pulse_sequence.to_dict()
    pulse_sequence_from_dict = isq.pulse.JaxBasedPulseSequence.from_dict(
        pulse_sequence_dict, pulses=[isq.utils.predefined.MultiDragPulseV2]*4
    )

    # Check that the waveform is the same after serialization and deserialization
    waveform_from_dict = pulse_sequence_from_dict.get_waveform(params)

    assert isinstance(waveform_from_dict, jnp.ndarray)

    assert jnp.allclose(waveform, waveform_from_dict)


def test_pulse_sequence_reader(tmp_path):

    pulse_sequence = isq.utils.predefined.get_multi_drag_pulse_sequence_v2()
    pulse_sequence.to_file(str(tmp_path))

    reader = isq.pulse.construct_pulse_sequence_reader(
        pulses=[isq.utils.predefined.MultiDragPulseV2]
    )

    reconstruct_pulse_sequence = reader(str(tmp_path))

    assert pulse_sequence == reconstruct_pulse_sequence
