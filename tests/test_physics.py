import pytest
import jax
import jax.numpy as jnp
from functools import partial
from qiskit_ibm_runtime.fake_provider import FakeJakartaV2  # type: ignore
import qiskit.quantum_info as qi
import inspeqtor as isq

jax.config.update("jax_enable_x64", True)


def test_signal_func_v3():
    qubit_info = isq.utils.predefined.get_mock_qubit_information()
    pulse_sequence = isq.utils.predefined.get_drag_pulse_sequence(qubit_info)
    dt = 2 / 9

    key = jax.random.PRNGKey(0)
    params = pulse_sequence.sample_params(key)

    signal = isq.physics.signal_func_v3(
        pulse_sequence.get_envelope,
        qubit_info.frequency,
        dt,
    )

    signal_params = isq.physics.SignalParameters(pulse_params=params, phase=0)

    discreted_signal = signal(
        signal_params, jnp.linspace(0, pulse_sequence.pulse_length_dt * dt, 100)
    )

    assert discreted_signal.shape == (100,)


def test_hamiltonian_fn():

    qubit_info = isq.utils.predefined.get_mock_qubit_information()
    pulse_sequence = isq.utils.predefined.get_drag_pulse_sequence(qubit_info)
    dt = 2 / 9

    key = jax.random.PRNGKey(0)
    params = pulse_sequence.sample_params(key)

    total_hamiltonian = isq.physics.gen_hamiltonian_from(
        qubit_informations=[qubit_info],
        coupling_constants=[],
        dims=2,
    )

    drive_terms = list(filter(isq.physics.pick_drive, total_hamiltonian))
    static_terms = list(filter(isq.physics.pick_static, total_hamiltonian))

    signals = {
        drive_term: isq.physics.signal_func_v3(
            pulse_sequence.get_envelope,
            drive_frequency=qubit_info.frequency,
            dt=dt,
        )
        for drive_term in drive_terms
    }

    signal_param = isq.physics.SignalParameters(pulse_params=params, phase=0)

    hamiltonian_args = {drive_terms[0]: signal_param}

    hamiltonian = partial(
        isq.physics.hamiltonian_fn,
        signals=signals,
        hamiltonian_terms=total_hamiltonian,
        static_terms=static_terms,
    )

    eval_hamiltonian = hamiltonian(hamiltonian_args, 1)

    assert eval_hamiltonian.shape == (2, 2)

    eval_hamiltonian = jax.vmap(hamiltonian, in_axes=(None, 0))(
        hamiltonian_args, jnp.array([1, 2])
    )

    assert eval_hamiltonian.shape == (2, 2, 2)


def test_run():

    batch_size = 10
    key = jax.random.PRNGKey(0)
    qubit_info = isq.data.QubitInformation(
        unit="GHz",
        qubit_idx=0,
        anharmonicity=-0.33,
        frequency=5.0,
        drive_strength=0.1,
    )
    dt = 2 / 9

    # Get the pulse sequence
    pulse_sequence = isq.utils.predefined.get_multi_drag_pulse_sequence_v2()
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
    simulator = isq.pennylane.get_simulator(qubit_info=qubit_info, t_eval=t_eval)

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

    qubit_info = isq.utils.predefined.get_mock_qubit_information()

    pulse_sequence = isq.utils.predefined.get_drag_pulse_sequence(qubit_info)

    key = jax.random.PRNGKey(0)
    params = pulse_sequence.sample_params(key)

    time_step = 2 / 9
    t_eval = jnp.arange(0, pulse_sequence.pulse_length_dt * time_step, step=time_step)

    # NOTE: This hamiltonian is the analytical rotated Hamiltonian of single qubit
    hamiltonian = partial(
        isq.utils.predefined.rotating_transmon_hamiltonian,
        qubit_info=qubit_info,
        signal=isq.physics.signal_func_v3(
            pulse_sequence.get_envelope, qubit_info.frequency, 2 / 9
        ),
    )

    jitted_simulator = jax.jit(
        partial(
            isq.physics.solver,
            t_eval=t_eval,
            hamiltonian=hamiltonian,
            y0=jnp.eye(2, dtype=jnp.complex64),
            t0=0,
            t1=pulse_sequence.pulse_length_dt * time_step,
        )
    )

    hamil_params = isq.physics.SignalParameters(pulse_params=params, phase=0)

    unitaries_manual_rotated = jitted_simulator(hamil_params)

    # NOTE: For the auto rotating test
    hamiltonian = partial(
        isq.utils.predefined.transmon_hamiltonian,  # This hamiltonian is in the lab frame
        qubit_info=qubit_info,
        signal=isq.physics.signal_func_v3(
            pulse_sequence.get_envelope, qubit_info.frequency, 2 / 9
        ),
    )

    frame = (jnp.pi * qubit_info.frequency) * isq.constant.Z

    rotating_hamiltonian = isq.physics.auto_rotating_frame_hamiltonian(
        hamiltonian, frame
    )

    jitted_simulator = jax.jit(
        partial(
            isq.physics.solver,
            t_eval=t_eval,
            hamiltonian=rotating_hamiltonian,
            y0=jnp.eye(2, dtype=jnp.complex64),
            t0=0,
            t1=pulse_sequence.pulse_length_dt * time_step,
        )
    )
    auto_rotated_unitaries = jitted_simulator(hamil_params)

    # NOTE: Crosscheck with pennylane
    qml_simulator = isq.pennylane.get_simulator(
        qubit_info=qubit_info,
        t_eval=t_eval,
        hamiltonian=isq.pennylane.rotating_transmon_hamiltonian,
    )
    qml_unitary = qml_simulator(pulse_sequence.get_waveform(params))

    # NOTE: Crosscheck with the hamiltonian_fn flow
    backend = FakeJakartaV2()
    backend_properties = isq.qiskit.IBMQDeviceProperties.from_backend(
        backend=backend, qubit_indices=[0]
    )
    backend_properties.qubit_informations = [qubit_info]

    coupling_infos = isq.physics.get_coupling_strengths(backend_properties)

    total_hamiltonian = isq.physics.gen_hamiltonian_from(
        qubit_informations=backend_properties.qubit_informations,
        coupling_constants=coupling_infos,
        dims=2,
    )

    drive_terms = list(filter(isq.physics.pick_drive, total_hamiltonian))
    static_terms = list(filter(isq.physics.pick_static, total_hamiltonian))

    signals = {
        drive_term: isq.physics.signal_func_v3(
            pulse_sequence.get_envelope,
            drive_frequency=backend_properties.qubit_informations[
                int(drive_term[-1])
            ].frequency,
            dt=backend_properties.dt,
        )
        for drive_term in drive_terms
    }

    hamiltonian = partial(
        isq.physics.hamiltonian_fn,
        signals=signals,
        hamiltonian_terms=total_hamiltonian,
        static_terms=static_terms,
    )

    hamiltonian_args = {
        drive_terms[0]: isq.physics.SignalParameters(
            pulse_params=params,
            phase=0.0,
        )
    }

    frame = total_hamiltonian[static_terms[0]].operator
    rotating_hamiltonian = isq.physics.auto_rotating_frame_hamiltonian(
        hamiltonian, frame
    )

    unitaries_hamiltonian_fn = isq.physics.solver(
        hamiltonian_args,
        t_eval=t_eval,
        hamiltonian=rotating_hamiltonian,
        y0=jnp.eye(2, dtype=jnp.complex64),
        t0=0,
        t1=pulse_sequence.pulse_length_dt * time_step,
    )

    unitaries_tuple = [
        (unitaries_manual_rotated, auto_rotated_unitaries),
        (unitaries_manual_rotated, qml_unitary),
        (qml_unitary, auto_rotated_unitaries),
        (unitaries_manual_rotated, unitaries_hamiltonian_fn),
    ]

    for uni_1, uni_2 in unitaries_tuple:
        fidelities = jax.vmap(isq.physics.gate_fidelity, in_axes=(0, 0))(uni_1, uni_2)

        assert jnp.allclose(fidelities, jnp.ones_like(fidelities), rtol=1e-4)


@pytest.mark.parametrize(
    "state",
    [
        jnp.array([[1, 0], [0, 0]]),
        jnp.array([[0, 0], [0, 1]]),
        jnp.array([[0.5, 0.5], [0.5, 0.5]]),
        jnp.array([[0.5, -0.5j], [0.5j, 0.5]]),
    ],
)
def test_state_tomography(state):

    exp_X = jnp.trace(state @ isq.constant.X)
    exp_Y = jnp.trace(state @ isq.constant.Y)
    exp_Z = jnp.trace(state @ isq.constant.Z)

    reconstructed_state = isq.physics.state_tomography(exp_X, exp_Y, exp_Z)

    assert jnp.allclose(state, reconstructed_state)

    # Check valid density matrix
    isq.physics.check_valid_density_matrix(reconstructed_state)


def wrong_to_superop(U: jnp.ndarray) -> jnp.ndarray:
    return jnp.kron(U, U.conj())


@pytest.mark.parametrize(
    "gate",
    [
        jnp.eye(2),
        isq.constant.X,
        isq.constant.Y,
        isq.constant.Z,
        # jax.scipy.linalg.sqrtm(isq.constant.X),
        isq.constant.SX,
    ],
)
def test_process_tomography(gate: jnp.ndarray):

    init_0 = jnp.array(isq.data.State.from_label("0", dm=True))
    init_1 = jnp.array(isq.data.State.from_label("1", dm=True))
    init_p = jnp.array(isq.data.State.from_label("+", dm=True))
    init_m = jnp.array(isq.data.State.from_label("r", dm=True))

    rho_0 = gate @ init_0 @ gate.conj().T
    rho_1 = gate @ init_1 @ gate.conj().T
    rho_p = gate @ init_p @ gate.conj().T
    rho_m = gate @ init_m @ gate.conj().T

    # Normally, those rho would be retrived from state tomography.
    assert jnp.allclose(
        isq.physics.avg_gate_fidelity_from_superop(
            isq.physics.process_tomography(rho_0, rho_1, rho_p, rho_m),
            isq.physics.to_superop(gate),
        ),
        jnp.array(1.0),
    )


def test_SX():

    state_0 = isq.data.State.from_label("0", dm=True)
    state_1 = isq.data.State.from_label("1", dm=True)
    SX = isq.constant.SX

    state_f = SX @ SX @ state_0 @ SX.conj().T @ SX.conj().T

    assert jnp.allclose(state_1, state_f)


from forest.benchmarking import operator_tools as ot  # type: ignore


@pytest.mark.parametrize(
    "gate_with_fidelity",
    [
        ([jnp.eye(2)], jnp.array(1.0)),
        ([isq.constant.X], jnp.array(1.0)),
        ([isq.constant.Y], jnp.array(1.0)),
        ([isq.constant.Z], jnp.array(1.0)),
        ([jax.scipy.linalg.sqrtm(isq.constant.X)], jnp.array(1.0)),
        ([isq.constant.SX], jnp.array(1.0)),
        (
            [
                jnp.array([[1, 0], [0, jnp.sqrt(1 - 0.2)]]),
                jnp.array([[0, jnp.sqrt(0.2)], [0, 0]]),
            ],
            isq.utils.helper.theoretical_fidelity_of_amplitude_dampling_channel_with_itself(
                0.2
            ),
        ),
    ],
)
def test_forest_process_tomography(gate_with_fidelity: list[jnp.ndarray]):

    gate = gate_with_fidelity[0]
    expected_fidelity = gate_with_fidelity[1]

    expvals = []
    for exp in isq.utils.predefined.default_expectation_values_order:
        rho_i = jnp.array(isq.data.State.from_label(exp.initial_state, dm=True))
        rho_f = ot.apply_kraus_ops_2_state(gate, rho_i)

        exp.expectation_value = jnp.trace(jnp.array(rho_f @ exp.observable_matrix)).real
        expvals.append(exp)

    est_superoperator = ot.choi2superop(isq.physics.forest_process_tomography(expvals))

    assert jnp.allclose(
        isq.physics.avg_gate_fidelity_from_superop(
            est_superoperator, ot.kraus2superop(gate)
        ),
        expected_fidelity,
    )


def test_direct_AFG_estimation_coefficients():

    # The order is [XX(+1), XX(-1), XY(+1), XY(-1), ...]
    expected_sx_coefficients = jnp.array(
        [
            1,
            -1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -1,
            1,
            0,
            0,
            1,
            -1,
            0,
            0,
        ]
    )

    coefficients = isq.model.direct_AFG_estimation_coefficients(isq.constant.SX)

    assert jnp.allclose(coefficients, expected_sx_coefficients)


def test_direct_AFG_estimation():

    target_unitary = jnp.array(qi.random_unitary(2).data)

    expvals = []
    for exp in isq.utils.predefined.default_expectation_values_order:
        expval = isq.model.calculate_exp(
            target_unitary, exp.observable_matrix, exp.initial_statevector
        )
        expvals.append(expval)

    coefficients = isq.model.direct_AFG_estimation_coefficients(target_unitary)

    AFG = isq.model.direct_AFG_estimation(coefficients, jnp.array(expvals))

    assert jnp.allclose(AFG, 1)
