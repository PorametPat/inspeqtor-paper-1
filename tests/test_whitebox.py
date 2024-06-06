import pytest
from specq_dev.jax import JaxBasedPulseSequence  # type: ignore
from specq_dev.shared import QubitInformation  # type: ignore
from specq_jax.pulse import MultiDragPulse
from specq_jax.pennylane import simulator as qmls
from specq_jax import core
import jax
import jax.numpy as jnp
import specq_jax.utils.testing as ut
import specq_jax.utils.helper as sqjh
from functools import partial
import specq_jax.simulator as sqjs
from specq_dev import qiskit as qk  # type: ignore
from qiskit_ibm_runtime.fake_provider import FakeJakartaV2  # type: ignore

jax.config.update("jax_enable_x64", True)


def test_signal_func_v3():
    qubit_info = ut.get_qubit_info()
    pulse_sequence = ut.get_drag_pulse_sequence(qubit_info)
    dt = 2 / 9

    key = jax.random.PRNGKey(0)
    params = pulse_sequence.sample_params(key)

    signal = sqjs.signal_func_v3(
        pulse_sequence.get_envelope,
        qubit_info.frequency,
        dt,
    )

    signal_params = sqjs.SignalParameters(pulse_params=params, phase=0)

    discreted_signal = signal(signal_params, jnp.linspace(0, pulse_sequence.pulse_length_dt * dt, 100))

    assert discreted_signal.shape == (100,)


def test_hamiltonian_fn():

    qubit_info = ut.get_qubit_info()
    pulse_sequence = ut.get_drag_pulse_sequence(qubit_info)
    dt = 2 / 9

    key = jax.random.PRNGKey(0)
    params = pulse_sequence.sample_params(key)

    total_hamiltonian = sqjs.gen_hamiltonian_from(
        qubit_informations=[qubit_info],
        coupling_constants=[],
        dims=2,
    )

    drive_terms = list(filter(sqjs.pick_drive, total_hamiltonian))
    static_terms = list(filter(sqjs.pick_static, total_hamiltonian))

    signals = {
        drive_term: sqjs.signal_func_v3(
            pulse_sequence.get_envelope,
            drive_frequency=qubit_info.frequency,
            dt=dt,
        )
        for drive_term in drive_terms
    }

    signal_param = sqjs.SignalParameters(pulse_params=params, phase=0)

    hamiltonian_args = {
        drive_terms[0]: signal_param
    }

    hamiltonian = partial(
        sqjs.hamiltonian_fn,
        signals=signals,
        hamiltonian_terms=total_hamiltonian,
        static_terms=static_terms,
    )

    eval_hamiltonian = hamiltonian(hamiltonian_args, 1)

    assert eval_hamiltonian.shape == (2, 2)

    eval_hamiltonian = jax.vmap(hamiltonian, in_axes=(None, 0))(hamiltonian_args, jnp.array([1, 2]))

    assert eval_hamiltonian.shape == (2, 2, 2)



def test_run():

    batch_size = 10
    key = jax.random.PRNGKey(0)
    qubit_info = QubitInformation(
        unit="GHz",
        qubit_idx=0,
        anharmonicity=-0.33,
        frequency=5.0,
        drive_strength=0.1,
    )
    dt = 2 / 9

    # Get the pulse sequence
    pulse_sequence = ut.get_multi_drag_pulse_sequence()
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
    simulator = qmls.get_simulator(qubit_info=qubit_info, t_eval=t_eval)

    # Solve for the unitaries
    # jit the simulator
    jitted_simulator = jax.jit(simulator)
    # batch the simulator
    batched_simulator = jax.vmap(jitted_simulator, in_axes=(0))
    # Get the unitaries
    unitaries = batched_simulator(waveforms)
    # Assert the unitaries shape
    assert unitaries.shape == (batch_size, pulse_sequence.pulse_length_dt, 2, 2)


def test_crosscheck_pennylane_difflax():

    qubit_info = ut.get_qubit_info()

    pulse_sequence = ut.get_drag_pulse_sequence(qubit_info)

    key = jax.random.PRNGKey(0)
    params = pulse_sequence.sample_params(key)

    time_step = 2 / 9
    t_eval = jnp.arange(0, pulse_sequence.pulse_length_dt * time_step, step=time_step)

    # NOTE: This hamiltonian is the analytical rotated Hamiltonian of single qubit
    hamiltonian = partial(
        sqjh.rotating_transmon_hamiltonian,
        qubit_info=qubit_info,
        signal=sqjs.signal_func_v3(
            pulse_sequence.get_envelope, qubit_info.frequency, 2 / 9
        ),
    )

    jitted_simulator = jax.jit(
        partial(
            sqjs.simulator,
            t_eval=t_eval,
            hamiltonian=hamiltonian,
            y0=jnp.eye(2, dtype=jnp.complex64),
        )
    )

    hamil_params = sqjs.SignalParameters(pulse_params=params, phase=0)

    unitaries_manual_rotated = jitted_simulator(hamil_params)

    # NOTE: For the auto rotating test
    hamiltonian = partial(
        sqjh.transmon_hamiltonian,  # This hamiltonian is in the lab frame
        qubit_info=qubit_info,
        signal=sqjs.signal_func_v3(
            pulse_sequence.get_envelope, qubit_info.frequency, 2 / 9
        ),
    )

    frame = (jnp.pi * qubit_info.frequency) * sqjh.Z

    rotating_hamiltonian = sqjs.auto_rotating_frame_hamiltonian(hamiltonian, frame)

    jitted_simulator = jax.jit(
        partial(
            sqjs.simulator,
            t_eval=t_eval,
            hamiltonian=rotating_hamiltonian,
            y0=jnp.eye(2, dtype=jnp.complex64),
        )
    )
    auto_rotated_unitaries = jitted_simulator(hamil_params)

    # NOTE: Crosscheck with pennylane
    qml_simulator = qmls.get_simulator(
        qubit_info=qubit_info,
        t_eval=t_eval,
        hamiltonian=qmls.rotating_transmon_hamiltonian,
    )
    qml_unitary = qml_simulator(pulse_sequence.get_waveform(params))

    # NOTE: Crosscheck with the hamiltonian_fn flow
    backend = FakeJakartaV2()
    backend_properties = qk.IBMQDeviceProperties.from_backend(
        backend=backend, qubit_indices=[0]
    )
    backend_properties.qubit_informations = [qubit_info]

    coupling_infos = sqjs.get_coupling_strengths(backend_properties)

    total_hamiltonian = sqjs.gen_hamiltonian_from(
        qubit_informations=backend_properties.qubit_informations,
        coupling_constants=coupling_infos,
        dims=2,
    )

    drive_terms = list(filter(sqjs.pick_drive, total_hamiltonian))
    static_terms = list(filter(sqjs.pick_static, total_hamiltonian))

    signals = {
        drive_term: sqjs.signal_func_v3(
            pulse_sequence.get_envelope,
            drive_frequency=backend_properties.qubit_informations[
                int(drive_term[-1])
            ].frequency,
            dt=backend_properties.dt,
        )
        for drive_term in drive_terms
    }

    hamiltonian = partial(
        sqjs.hamiltonian_fn,
        signals=signals,
        hamiltonian_terms=total_hamiltonian,
        static_terms=static_terms,
    )

    hamiltonian_args = {
        drive_terms[0]: sqjs.SignalParameters(
            pulse_params=params,
            phase=0.0,
        )
    }

    frame = total_hamiltonian[static_terms[0]].operator
    rotating_hamiltonian = sqjs.auto_rotating_frame_hamiltonian(hamiltonian, frame)

    unitaries_hamiltonian_fn = sqjs.simulator(
        hamiltonian_args,
        t_eval=t_eval,
        hamiltonian=rotating_hamiltonian,
        y0=jnp.eye(2, dtype=jnp.complex64),
    )

    unitaries_tuple = [
        (unitaries_manual_rotated, auto_rotated_unitaries),
        (unitaries_manual_rotated, qml_unitary),
        (qml_unitary, auto_rotated_unitaries),
        (unitaries_manual_rotated, unitaries_hamiltonian_fn),
    ]

    for uni_1, uni_2 in unitaries_tuple:
        fidelities = jax.vmap(core.gate_fidelity, in_axes=(0, 0))(uni_1, uni_2)

        assert jnp.allclose(fidelities, jnp.ones_like(fidelities), rtol=1e-4)
