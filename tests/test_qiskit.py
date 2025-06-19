from qiskit_ibm_runtime.fake_provider import FakeJakartaV2  # type: ignore
import jax
import jax.numpy as jnp
from functools import partial
from qiskit_ibm_runtime import SamplerV2  # type: ignore
import inspeqtor as isq


def test_execute_experiment_using_fake_backend_v2():

    EXPERIMENT_IDENTIFIER = "0021"
    SHOTS = 3000
    EXPERIMENT_TAGS = [EXPERIMENT_IDENTIFIER, "random_drag", "MCMD", "test_2"]
    SAMPLE_SIZE = 1
    QUBIT_IDX = 0
    INSTANCE = "ibm_test"

    # service, backend = qk.get_ibm_service_and_backend(instance=INSTANCE, backend_name=BACKEND_NAME)

    backend = FakeJakartaV2()
    service = None

    backend_properties = isq.qiskit.IBMQDeviceProperties.from_backend(
        backend=backend, qubit_indices=[QUBIT_IDX]
    )

    drive_str = backend_properties.qubit_informations[0].drive_strength
    dt = backend_properties.dt
    total_length = 80
    amp = 0.8
    area = 1 / (2 * drive_str * amp) / dt
    sigma = (1 * area) / (amp * jnp.sqrt(2 * jnp.pi))

    pulse_sequence = isq.utils.predefined.get_multi_drag_pulse_sequence_v2()

    config = isq.data.ExperimentConfiguration(
        qubits=backend_properties.qubit_informations,
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

    # options = options_v2 # if backend_properties.is_simulator else options_v1
    options = isq.qiskit.make_sampler_options(shots=SHOTS)

    execute_dataframe, jobs = isq.qiskit.execute_experiment(
        df, backend_properties, options, transpiled_circuits, service, SamplerV2
    )

    _jobs, is_all_done, is_error, num_done = isq.qiskit.check_jobs_status(
        execute_dataframe, service=service, jobs=jobs
    )

    result_df = isq.qiskit.get_result_from_remote(
        execute_dataframe, jobs, isq.qiskit.extract_results_v2
    )

    exp_data = isq.data.ExperimentData(
        experiment_config=config,
        preprocess_data=result_df,
    )


test_cases = [
    isq.qiskit.CompackConfig(identifier="0001", shots=4096, sample_size=1, qubit_idx=0),
    isq.qiskit.CompackConfig(identifier="0002", shots=4096, sample_size=1, qubit_idx=1),
]


def test_execute_experiment_using_fake_backend_v3():

    INSTANCE = "ibm_test"
    backend = FakeJakartaV2()
    service = None

    qubit_indices = [test_case.qubit_idx for test_case in test_cases]

    backend_properties = isq.qiskit.IBMQDeviceProperties.from_backend(
        backend=backend, qubit_indices=qubit_indices
    )

    pulse_sequence = isq.utils.predefined.get_multi_drag_pulse_sequence_v2()

    # Initialize configutations
    configs = []
    for compact_config in test_cases:
        config = isq.data.ExperimentConfiguration(
            qubits=[backend_properties.qubit_informations[compact_config.qubit_idx]],
            expectation_values_order=isq.utils.predefined.default_expectation_values_order,
            parameter_names=pulse_sequence.get_parameter_names(),
            backend_name=backend_properties.name,
            shots=compact_config.shots,
            EXPERIMENT_IDENTIFIER=compact_config.identifier,
            EXPERIMENT_TAGS=[compact_config.identifier],
            description="The experiment to test random drag sequence",
            device_cycle_time_ns=backend_properties.dt,
            sequence_duration_dt=pulse_sequence.pulse_length_dt,
            instance=INSTANCE,
            sample_size=compact_config.sample_size,
        )
        configs.append(config)

    keys = jax.random.split(jax.random.PRNGKey(0), len(test_cases))
    keys = [key for key in keys]  # NOTE: Not necessary but will satisfy Mypy

    # Prepare the dataframes and generate quantum circuits
    prepared_dfs, qcs = isq.qiskit.prepare_parallel_experiment_v2(
        configs,
        [pulse_sequence] * len(test_cases),
        keys,
        backend_properties,
    )

    for df, config in zip(prepared_dfs, configs, strict=True):
        assert len(qcs) == int(
            config.sample_size
            * len(isq.utils.predefined.default_expectation_values_order)
        )

        assert len(df) == len(qcs)

    # Transpile the circuits
    tqcs = isq.qiskit.transpile_circuits_v2(
        qcs,
        backend_properties,
        [qubit.qubit_idx for qubit in backend_properties.qubit_informations],
    )

    assert len(tqcs) == len(qcs)

    # Execute the experiments
    options_v3 = isq.qiskit.make_sampler_options(shots=configs[0].shots)
    executed_dfs, jobs = isq.qiskit.execute_parallel_experiment(
        prepared_dfs, backend_properties, options_v3, tqcs, service
    )

    # Check the jobs status
    _jobs, is_all_done, is_error, num_done = isq.qiskit.check_jobs_status(
        executed_dfs[0], service=service, jobs=jobs
    )

    # Retrieve the experiments
    result_dfs = [
        isq.qiskit.get_result_from_remote(
            executed_dfs[qubit.qubit_idx],
            jobs,
            partial(isq.qiskit.extract_results_v3, qubit_idx=qubit.qubit_idx),
        )
        for qubit in backend_properties.qubit_informations
    ]

    assert len(result_dfs) == len(test_cases)

    # Try to save them
    for config, result_df in zip(configs, result_dfs):
        exp_data = isq.data.ExperimentData(
            experiment_config=config, preprocess_data=result_df
        )
