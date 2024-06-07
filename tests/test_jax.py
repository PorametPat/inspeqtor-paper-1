import jax
jax.config.update("jax_enable_x64", True)

from qiskit_ibm_runtime.fake_provider import FakeJakartaV2  # type: ignore
from specq_dev import qiskit as qk, shared  # type: ignore
from specq_dev.test_utils import get_jax_based_pulse_sequence  # type: ignore
import jax.numpy as jnp

from qiskit_ibm_runtime import SamplerV2, SamplerV1, Options  # type: ignore

from specq_jax import core, data, model as specq_model, simulator as sqjs  # type: ignore
import specq_jax.utils.testing as ut
import specq_jax.utils.helper as sqjh

from torch import Generator
from torch.utils.data import DataLoader, Subset
import optax  # type: ignore
from dataclasses import asdict
import logging

def test_pulse_reader(tmp_path):

    pulse_sequence = get_jax_based_pulse_sequence()
    pulse_sequence.to_file(str(tmp_path))

    reread_pulse_sequence = data.pulse_reader(str(tmp_path))

    assert reread_pulse_sequence == pulse_sequence

# @pytest.mark.skip(reason="Need to fix the test")
def test_end_to_end(tmp_path):

    d = tmp_path / "data"
    d.mkdir()

    EXPERIMENT_IDENTIFIER = "0021"
    SHOTS = 3000
    EXPERIMENT_TAGS = [EXPERIMENT_IDENTIFIER, "random_drag", "MCMD", "test_2"]
    SAMPLE_SIZE = 10
    QUBIT_IDX = 0
    INSTANCE = "ibm_test"

    # service, backend = qk.get_ibm_service_and_backend(instance=INSTANCE, backend_name=BACKEND_NAME)

    backend = FakeJakartaV2()
    service = None

    backend_properties = qk.IBMQDeviceProperties.from_backend(
        backend=backend, qubit_indices=[QUBIT_IDX]
    )

    qubit_info = backend_properties.qubit_informations[QUBIT_IDX]

    logging.debug(f"Backend properties: {backend_properties}")

    pulse_sequence = ut.get_multi_drag_pulse_sequence_v2()
    pulse_sequence.to_file(str(d))

    config = shared.ExperimentConfiguration(
        qubits=[qubit_info],
        expectation_values_order=shared.default_expectation_values,
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

    df, circuits = qk.prepare_experiment(
        config, pulse_sequence, key, backend_properties
    )

    assert len(circuits) == int(
        config.sample_size * len(shared.default_expectation_values)
    )

    assert len(df) == len(circuits)

    transpiled_circuits = qk.transpile_circuits(
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

    Sampler = SamplerV2 if backend_properties.is_simulator else SamplerV1
    options = options_v2 if backend_properties.is_simulator else options_v1

    execute_dataframe, jobs = qk.execute_experiment(
        df, backend_properties, options, transpiled_circuits, service, Sampler
    )

    _jobs, is_all_done, is_error, num_done = qk.check_jobs_status(
        execute_dataframe, service=service, jobs=jobs
    )

    result_df = qk.get_result_from_remote(
        execute_dataframe, 
        jobs,
        qk.extract_results_v2
    )

    exp_data = shared.ExperimentData(
        experiment_config=config,
        preprocess_data=result_df,
    )

    # Test the save and load

    exp_data.save_to_folder(path=str(d))

    exp_data, pulse_parameters, unitaries, expectations, pulse_sequence, simulator = (
        data.load_data(
            d,
        )
    )
    # NOTE: flatten the pulse_parameters
    pulse_parameters = pulse_parameters.reshape(SAMPLE_SIZE, -1)

    # start_idx, end_idx = 0, SAMPLE_SIZE
    # batch_size = 2
    MASTER_KEY_SEED = 0
    # g = Generator()
    # g.manual_seed(MASTER_KEY_SEED)

    master_key = jax.random.PRNGKey(MASTER_KEY_SEED)
    random_split_key, model_key, gate_optim_key, dropout_key = jax.random.split(master_key, 4)

    # # Final goal of setting up is to create a dataset and a dataloader
    # dataset = data.SpecQDataset(
    #     pulse_parameters=pulse_parameters[start_idx:end_idx],
    #     unitaries=unitaries[start_idx:end_idx],
    #     expectation_values=expectations[start_idx:end_idx],
    # )

    # # Randomly split dataset into training and validation
    # val_indices = jax.random.choice(
    #     random_split_key, len(dataset), (int(0.2 * len(dataset)),), replace=False
    # ).tolist()

    # training_indices = list(set([i for i in range(len(dataset))]) - set(val_indices))

    # train_dataset = Subset(dataset, training_indices)
    # val_dataset = Subset(dataset, val_indices)

    # train_dataloader = DataLoader(
    #     train_dataset, batch_size=batch_size, shuffle=True, generator=g
    # )
    # val_dataloader = DataLoader(
    #     val_dataset, batch_size=batch_size, shuffle=True, generator=g
    # )

    model = specq_model.BasicBlackBox(feature_size=5)

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

    # train_step, test_step, model_params, opt_state, transform_state = core.create_train_step(
    #     key=model_key,
    #     model=model,
    #     optimiser=optimiser,
    #     # loss_fn=lambda params, pulse_parameters, unitaries, expectations, training: core.loss(
    #     #     params, pulse_parameters, unitaries, expectations, training, model
    #     # ),
    #     input_shape=(batch_size, pulse_parameters.shape[1]),
    # )

    # model_params, opt_state, history = core.with_validation_train(
    #     train_dataloader,
    #     val_dataloader,
    #     train_step,
    #     test_step,
    #     model_params,
    #     opt_state,
    #     dropout_key,
    #     num_epochs=2,
    #     transform=None,
    #     transform_state=transform_state,
    # )

    transform = core.default_adaptive_lr_transform(
        PATIENCE=10,
        COOLDOWN=0,
        FACTOR=0.1,
        RTOL=1e-4,
        ACCUMULATION_SIZE=2,
    )

    train_dataloader, val_dataloader = data.prepare_dataset(
        pulse_parameters=pulse_parameters,
        unitaries=unitaries,
        expectation_values=expectations,
        key=random_split_key,
    )

    model_params, opt_state, history = core.train_model(
        key=model_key,
        model=model,
        optimiser=optimiser,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epoch=2,
        transform=transform,
    )


    MODEL_PATH = tmp_path / "ckpts/"
    MODEL_PATH.mkdir()

    model_path = data.save_model(
        str(MODEL_PATH),
        "0020",
        pulse_sequence,
        sqjh.rotating_transmon_hamiltonian,
        asdict(model),
        model_params,
        history,
        with_auto_datetime=False,
    )

    target_unitary = jax.scipy.linalg.sqrtm(core.X)
    fun = lambda x: core.gate_loss(
        x,
        model,
        model_params,
        simulator,
        pulse_sequence,
        target_unitary,
    )

    lower, upper = pulse_sequence.get_bounds()
    lower = pulse_sequence.list_of_params_to_array(lower)
    upper = pulse_sequence.list_of_params_to_array(upper)

    pulse_params = pulse_sequence.sample_params(gate_optim_key)
    x0 = pulse_sequence.list_of_params_to_array(pulse_params)

    opt_params, state = core.optimize(x0, lower, upper, fun)

    # Plot the optmized result
    opt_pulse_params = pulse_sequence.array_to_list_of_params(opt_params)

    # Calculate the expectation values
    waveform_opt = pulse_sequence.get_waveform(opt_pulse_params)
    opt_signal_params = sqjs.SignalParameters(
        pulse_params=opt_pulse_params,
        phase=0
    )
    unitaries_opt = simulator(opt_signal_params)

    Wos_predicted_dict = core.evalulate_model_to_pauli(
        model, model_params, jnp.expand_dims(opt_params, 0)
    )

    performance = {
        "X": float(Wos_predicted_dict["X"][0]),
        "Y": float(Wos_predicted_dict["Y"][0]),
        "Z": float(Wos_predicted_dict["Z"][0]),
        "gate": float(core.gate_fidelity(unitaries_opt[-1], target_unitary)),
    }

    print("Benchmarking the gate")

    qc = qk.get_circuit(
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

    # qpt = ProcessTomography(qc, backend=backend, physical_qubits=[0])

    # print("Executing process tomography experiment")

    # qpt_job = qpt.run(backend=backend)
    # qpt_results = qpt_job.block_for_results()

    # qpt_value = qpt_results.analysis_results(0).value

    # fidelity_qpt_predicted = average_gate_fidelity(
    #     qpt_value, Operator(np.array(unitaries_opt[-1]))
    # )
    # fidelity_qpt_target = average_gate_fidelity(
    #     qpt_value, Operator(np.array(target_unitary))
    # )

    # performance["fidelity_qpt_predicted"] = fidelity_qpt_predicted
    # performance["fidelity_qpt_target"] = fidelity_qpt_target

    # print("Save the process tomography data to the folder")

    # # Save in analysis folder using numpy
    # os.makedirs("analysis", exist_ok=True)

    # try:
    #     np.save("analysis/qpt_value.npy", qpt_value)
    # except Exception as e:
    #     print(e)
    #     print("Failed to save the QPT data")

    # # Save the performance to the analysis folder in json
    # try:
    #     with open("analysis/performance.json", "w") as f:
    #         json.dump(performance, f)
    # except Exception as e:
    #     print(e)
    #     print("Failed to save the performance data")
    
