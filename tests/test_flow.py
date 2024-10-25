import pytest
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial
from qiskit_ibm_runtime import SamplerV2, Options  # type: ignore
from qiskit_ibm_runtime.fake_provider import FakeJakartaV2  # type: ignore
import logging
import optax  # type: ignore
from dataclasses import asdict

import inspeqtor as isq


# @pytest.mark.skip(reason="This is an end-to-end test")
def test_end_to_end(tmp_path):

    d = tmp_path / "data"
    d.mkdir()

    EXPERIMENT_IDENTIFIER = "0021"
    SHOTS = 3000
    EXPERIMENT_TAGS = [EXPERIMENT_IDENTIFIER, "random_drag", "MCMD", "test_2"]
    SAMPLE_SIZE = 10
    QUBIT_IDX = 0
    INSTANCE = "ibm_test"

    # service, backend = isq.qiskit.get_ibm_service_and_backend(instance=INSTANCE, backend_name=BACKEND_NAME)

    backend = FakeJakartaV2()
    service = None

    backend_properties = isq.qiskit.IBMQDeviceProperties.from_backend(
        backend=backend, qubit_indices=[QUBIT_IDX]
    )

    qubit_info = backend_properties.qubit_informations[QUBIT_IDX]

    logging.debug(f"Backend properties: {backend_properties}")

    pulse_sequence = isq.utils.predefined.get_multi_drag_pulse_sequence_v2()
    pulse_sequence.to_file(str(d))

    config = isq.data.ExperimentConfiguration(
        qubits=[qubit_info],
        expectation_values_order=isq.utils.predefined.default_expectation_values_order,
        parameter_names=pulse_sequence.get_parameter_names(),
        backend_name=backend_properties.name,
        shots=SHOTS,
        EXPERIMENT_IDENTIFIER=EXPERIMENT_IDENTIFIER,
        EXPERIMENT_TAGS=EXPERIMENT_TAGS,
        description="The experiment to test random drag sequence",
        device_cycle_time_ns=backend_properties.dt,
        sequence_duration_dt=pulse_sequence.pulse_length_dt,
        instance=INSTANCE,
        sample_size=SAMPLE_SIZE,
    )

    key = jax.random.PRNGKey(0)

    df, circuits = isq.qiskit.prepare_experiment(
        config, pulse_sequence, key, backend_properties
    )

    assert len(circuits) == int(
        config.sample_size * len(isq.utils.predefined.default_expectation_values_order)
    )

    assert len(df) == len(circuits)

    transpiled_circuits = isq.qiskit.transpile_circuits(
        circuits, backend_properties.backend_instance, inital_layout=[QUBIT_IDX]
    )

    options_v1 = Options(
        optimization_level=0,
        resilience_level=0,
        execution=dict(  # type: ignore
            shots=SHOTS,
        ),
        transpilation=dict(  # type: ignore
            skip_transpilation=True,
        ),
    )

    options_v2 = dict(
        default_shots=SHOTS,
        dynamical_decoupling=dict(enable=False),
        twirling=dict(enable_gates=False, enable_measure=False),
    )

    Sampler = SamplerV2 
    options = options_v2 if backend_properties.is_simulator else options_v1

    execute_dataframe, jobs = isq.qiskit.execute_experiment(
        df, backend_properties, options, transpiled_circuits, service, Sampler
    )

    _jobs, is_all_done, is_error, num_done = isq.qiskit.check_jobs_status(
        execute_dataframe, service=service, jobs=jobs
    )

    extract_func = partial(isq.qiskit.extract_results_v3, qubit_idx=QUBIT_IDX)

    result_df = isq.qiskit.get_result_from_remote(
        execute_dataframe,
        jobs,
        isq.qiskit.extract_results_v2,  # TODO: change to v3?
        # extract_func
    )

    exp_data = isq.data.ExperimentData(
        experiment_config=config,
        preprocess_data=result_df,
    )

    # Test the save and load

    exp_data.save_to_folder(path=str(d))

    # exp_data, pulse_parameters, unitaries, expectations, pulse_sequence, simulator = (
    #     isq.utils.helper.load_data_from_path(
    #         d,
    #     )
    # )

    loaded_data = isq.utils.helper.load_data_from_path(
            d,
        )

    exp_data = loaded_data.experiment_data
    pulse_parameters = loaded_data.pulse_parameters
    unitaries = loaded_data.unitaries
    expectations = loaded_data.expectation_values
    pulse_sequence = loaded_data.pulse_sequence
    simulator = loaded_data.whitebox


    # NOTE: flatten the pulse_parameters
    pulse_parameters = pulse_parameters.reshape(SAMPLE_SIZE, -1)

    # Test the pulse parameter conversion
    sample_pulse_params = pulse_parameters[0]
    recovered_pulse_params = isq.pulse.array_to_list_of_params(
        sample_pulse_params, pulse_sequence.get_parameter_names()
    )

    from_exp_data_sample_pulse_params = exp_data.get_parameters_dict_list()[0]

    assert recovered_pulse_params == from_exp_data_sample_pulse_params

    MASTER_KEY_SEED = 0

    master_key = jax.random.PRNGKey(MASTER_KEY_SEED)
    random_split_key, model_key, gate_optim_key, dropout_key = jax.random.split(
        master_key, 4
    )

    model = isq.model.BasicBlackBox(feature_size=5)

    warmup_start_lr, warmup_steps = 1e-6, 10
    start_lr, end_lr, steps = 1e-2, 1e-5, 100
    lr_scheduler = optax.join_schedules(
        [
            optax.linear_schedule(
                warmup_start_lr,
                start_lr,
                warmup_steps,
            ),
            optax.linear_schedule(
                start_lr,
                end_lr,
                steps - warmup_steps,
            ),
        ],
        [warmup_steps],
    )

    optimiser = optax.adam(lr_scheduler)

    transform = isq.model.default_adaptive_lr_transform(
        PATIENCE=10,
        COOLDOWN=0,
        FACTOR=0.1,
        RTOL=1e-4,
        ACCUMULATION_SIZE=2,
    )

    train_dataloader, val_dataloader, test_dataloader = isq.data.prepare_dataset(
        pulse_parameters=pulse_parameters,
        unitaries=unitaries,
        expectation_values=expectations,
    )

    model_params, opt_state, history = isq.model.train_model(
        key=model_key,
        model=model,
        optimiser=optimiser,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        num_epoch=2,
        lr_transformer=transform,
    )

    MODEL_PATH = tmp_path / "ckpts/"
    MODEL_PATH.mkdir()

    model_path = isq.model.save_model(
        str(MODEL_PATH),
        "0020",
        pulse_sequence,
        isq.utils.predefined.rotating_transmon_hamiltonian,
        asdict(model),
        model_params,
        history,
        with_auto_datetime=False,
    )

    parameter_structure = pulse_sequence.get_parameter_names()
    array_to_list_of_params = lambda x: isq.pulse.array_to_list_of_params(
        x, parameter_structure
    )

    target_unitary = jax.scipy.linalg.sqrtm(isq.constant.X)
    fun = lambda x: isq.model.gate_loss_v2(
        x,
        model,
        model_params,
        simulator,
        array_to_list_of_params,
        target_unitary,
    )

    _lower, _upper = pulse_sequence.get_bounds()
    lower = isq.pulse.list_of_params_to_array(
        _lower, parameter_structure
    )
    upper = isq.pulse.list_of_params_to_array(
        _upper, parameter_structure
    )

    pulse_params = pulse_sequence.sample_params(gate_optim_key)
    x0 = isq.pulse.list_of_params_to_array(
        pulse_params, parameter_structure
    )

    opt_params, state = isq.optimizer.optimize(x0, lower, upper, fun, 10)

    # Plot the optmized result
    opt_pulse_params = isq.pulse.array_to_list_of_params(
        opt_params, parameter_structure
    )

    # Calculate the expectation values
    waveform_opt = pulse_sequence.get_waveform(opt_pulse_params)
    opt_signal_params = isq.physics.SignalParameters(
        pulse_params=opt_pulse_params, phase=0
    )
    unitaries_opt = simulator(opt_signal_params)

    Wos_predicted_dict = isq.model.evalulate_model_to_pauli_avg_gate_fidelity(
        model, model_params, jnp.expand_dims(opt_params, 0)
    )

    performance = {
        "X": float(Wos_predicted_dict["X"][0]),
        "Y": float(Wos_predicted_dict["Y"][0]),
        "Z": float(Wos_predicted_dict["Z"][0]),
        "gate": float(
            isq.physics.avg_gate_fidelity_from_superop(
                isq.physics.to_superop(unitaries_opt[-1]),
                isq.physics.to_superop(target_unitary),
            )
        ),
    }

    print("Benchmarking the gate")

    qc = isq.qiskit.get_circuit(
        initial_state="0",
        waveforms=waveform_opt,
        observable="Z",
        backend=backend_properties,
        q_idx=0,
        add_pulse=True,
        change_basis=False,
        add_measure=False,
        enable_MCMD=False,
    )

    logging.info("Skip the process tomography for now")
