
from qiskit_ibm_runtime.fake_provider import FakeJakartaV2  # type: ignore
from specq_dev import qiskit as specq_qiskit, shared # type: ignore
from specq_dev.jax import JaxBasedPulseSequence # type: ignore
from specq_dev.test_utils import get_jax_based_pulse_sequence # type: ignore
import jax
import jax.numpy as jnp

from qiskit_ibm_runtime import SamplerV2, SamplerV1, Options  # type: ignore

from specq_jax import core, data, model as specq_model  # type: ignore

from torch import Generator
from torch.utils.data import DataLoader, Subset
import optax # type: ignore
from dataclasses import asdict
import logging

from pathlib import Path

def test_end_to_end(tmp_path):

    EXPERIMENT_IDENTIFIER = "0021"
    SHOTS = 3000
    EXPERIMENT_TAGS = [EXPERIMENT_IDENTIFIER, "random_drag", "MCMD", "test_2"]
    SAMPLE_SIZE = 10
    QUBIT_IDX = 0
    INSTANCE = "ibm_test"

    # service, backend = specq_qiskit.get_ibm_service_and_backend(instance=INSTANCE, backend_name=BACKEND_NAME)

    backend = FakeJakartaV2()
    service = None

    backend_properties = specq_qiskit.IBMQDeviceProperties.from_backend(
        backend=backend, qubit_indices=[QUBIT_IDX]
    )

    qubit_info = specq_qiskit.get_qubit_information_from_backend(
        backend, 0, service=service
    )

    logging.debug(f"Backend properties: {backend_properties}")

    pulse_sequence = get_jax_based_pulse_sequence()

    config = shared.ExperimentConfigV3(
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

    df, circuits = specq_qiskit.prepare_experiment(
        config, pulse_sequence, key, backend_properties
    )

    assert len(circuits) == int(
        config.sample_size * len(shared.default_expectation_values)
    )

    assert len(df) == len(circuits)

    transpiled_circuits = specq_qiskit.transpile_circuits(
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

    execute_dataframe, jobs = specq_qiskit.execute_experiment(
        df, backend_properties, options, transpiled_circuits, service, Sampler
    )

    _jobs, is_all_done, is_error = specq_qiskit.check_jobs_status(execute_dataframe, service=service, jobs=jobs)

    result_df = specq_qiskit.get_result_from_remote(execute_dataframe, jobs)


    exp_data = shared.ExperimentDataV3(
        experiment_config=config,
        preprocess_data=result_df,
    )

    # Test the save and load
    d = tmp_path / "data"
    d.mkdir()
    exp_data.save_to_folder(path=str(d))

    exp_data, pulse_parameters, unitaries, expectations, pulse_sequence, simulator = (
        data.load_data(
            str(d),
            get_jax_based_pulse_sequence,
            core.rotating_transmon_hamiltonian,
        )
    )
    # NOTE: flatten the pulse_parameters
    pulse_parameters = pulse_parameters.reshape(SAMPLE_SIZE, -1)

    start_idx, end_idx = 0, SAMPLE_SIZE
    batch_size = 2
    MASTER_KEY_SEED = 0
    g = Generator()
    g.manual_seed(MASTER_KEY_SEED)

    master_key = jax.random.PRNGKey(MASTER_KEY_SEED)
    random_split_key, model_key, gate_optim_key = jax.random.split(master_key, 3)

    # Final goal of setting up is to create a dataset and a dataloader
    dataset = core.SpecQDataset(
        pulse_parameters=pulse_parameters[start_idx: end_idx],
        unitaries=unitaries[start_idx: end_idx],
        expectation_values=expectations[start_idx: end_idx],
    )

    # Randomly split dataset into training and validation
    val_indices = jax.random.choice(
        random_split_key, len(dataset), (int(0.2 * len(dataset)),), replace=False
    ).tolist()

    training_indices = list(
        set([i for i in range(len(dataset))]) - set(val_indices)
    )

    train_dataset = Subset(dataset, training_indices)
    val_dataset = Subset(dataset, val_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, generator=g)

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

    train_step, test_step, model_params, opt_state = core.create_train_step(
        key=model_key,
        model=model,
        optimiser=optimiser,
        loss_fn=lambda params, pulse_parameters, unitaries, expectations: core.loss(
            params, pulse_parameters, unitaries, expectations, model
        ),
        input_shape=(batch_size, pulse_parameters.shape[1]),
    )

    model_params, opt_state, history = core.with_validation_train(
        train_dataloader,
        val_dataloader,
        train_step,
        test_step,
        model_params,
        opt_state,
    )

    MODEL_PATH = tmp_path / "ckpts/"
    MODEL_PATH.mkdir()

    _path = data.save_model(
        str(MODEL_PATH),
        "0020",
        pulse_sequence,
        core.rotating_transmon_hamiltonian,
        asdict(model),
        model_params,
        history,
        with_auto_datetime=False,
    )