import warnings

warnings.filterwarnings("ignore", category=UserWarning)  # intercept UserWarnings

import jax

jax.config.update("jax_enable_x64", True)  # enable 64-bit precision
warnings.filterwarnings("ignore", category=jax._src.util.NumpyComplexWarning)

import jax.numpy as jnp
import numpy as np
from dataclasses import asdict
import json
import typer
from rich.status import Status
import time
from functools import partial
from pathlib import Path
import typing
import pandas as pd  # type: ignore
from qiskit.primitives import PrimitiveJob  # type: ignore
from qiskit_ibm_runtime import RuntimeJobV2  # type: ignore
from qiskit.providers import JobStatus  # type: ignore
from qiskit_ibm_runtime import SamplerV2, Session  # type: ignore
import optax  # type: ignore
import seaborn as sns
import matplotlib.pyplot as plt

import inspeqtor as sq
from receipt_utils import (
    read_config,
    BASE_THEME,
    console,
    get_backend_properies,
    target_gate_choices,
)


def step_01_prepare_parallel_experiment(
    paths: list[Path],
    backend_properties: sq.qiskit.IBMQDeviceProperties,
    status: typing.Union[Status, None] = None,
):

    # Read the config
    configs: list[sq.data.ExperimentConfiguration] = [
        read_config(sub_dir) for sub_dir in paths
    ]
    pulse_sequences = [sq.utils.helper.pulse_reader(str(sub_dir)) for sub_dir in paths]

    # Get the key from the configurations
    pulse_keys = []
    for config in configs:
        all_keys = sq.utils.helper.generate_keys(
            int(config.additional_info["MASTER_KEY_SEED"])
        )
        pulse_keys.append(all_keys.pulse_key)

    if status is not None:
        status.update(
            BASE_THEME + "Preparing the template of dataframe for experiment data..."
        )

    # Prepare the dataframes and generate quantum circuits
    prepared_dfs, qcs = sq.qiskit.prepare_parallel_experiment_v2(
        configs,
        pulse_sequences,
        pulse_keys,
        backend_properties,
        enable_MCMD=False,
    )

    console.log("The template of dataframe for experiment data is prepared")
    if status is not None:
        status.update(BASE_THEME + "Transpiling quantum circuits...")

    transpiled_circuits = sq.qiskit.transpile_circuits_v2(
        quantum_circuits=qcs,
        backend_properties=backend_properties,
        inital_layout=[
            qubit.qubit_idx for qubit in backend_properties.qubit_informations
        ],
    )

    console.log("The quantum circuits are transpiled")
    if status is not None:
        status.update(BASE_THEME + "Saving the prepared data to folder...")

    for i, path, config, df in zip(range(len(paths)), paths, configs, prepared_dfs):
        df.to_csv(path / "predefine_exp_data.csv", index=False)

    if status is not None:
        status.update(BASE_THEME + "Saving the transpiled circuits to folder...")

    # Save the quantum circuits to the sqd.shared directory
    sq.qiskit.save_quantum_circuits(
        transpiled_circuits, str(paths[0].parent / "transpiled_circuits.qpy")
    )

    console.log("The prepared data and transpiled circuits are saved to the folder")


def step_02_execute_parallel_experiment(
    sub_dirs: list[Path],
    status: typing.Union[Status, None] = None,
):
    configs = [read_config(sub_dir) for sub_dir in sub_dirs]

    BACKEND_NAME = configs[0].backend_name
    INSTANCE = configs[0].instance

    qubit_informations = [config.qubits for config in configs]
    # Flatten the list
    qubit_indices = [
        item.qubit_idx for sublist in qubit_informations for item in sublist
    ]

    service, backend_properties = get_backend_properies(
        BACKEND_NAME, INSTANCE, qubit_indices, status=status
    )

    # Read the prepared data
    prepared_dfs = [
        pd.read_csv(sub_dir / "predefine_exp_data.csv") for sub_dir in sub_dirs
    ]

    # Read the transpiled circuits
    transpiled_circuits = sq.qiskit.load_quantum_circuits(
        sub_dirs[0].parent / "transpiled_circuits.qpy"
    )

    if status is not None:
        status.update(BASE_THEME + "Running the experiment on the backend...")

    # Execute the experiments
    options = sq.qiskit.make_sampler_options(shots=configs[0].shots)
    executed_dfs, jobs = sq.qiskit.execute_parallel_experiment(
        dataframes=prepared_dfs,
        backend_properties=backend_properties,
        options=options,
        transpiled_circuits=transpiled_circuits,
        service=service,
    )

    if status is not None:
        status.update(BASE_THEME + "Saving the job_id to the dataframe...")

    for i, path, df in zip(range(len(sub_dirs)), sub_dirs, executed_dfs):
        df.to_csv(path / "executed_exp_data.csv", index=False)

    console.log("The job_id is saved to the dataframe")

    return jobs


def step_03_wait_for_jobs(
    path: Path,
    wait_interval: int = 10,
    jobs: typing.Union[list[typing.Union[PrimitiveJob, RuntimeJobV2]], None] = None,
    status: typing.Union[Status, None] = None,
):

    # Get the service and backend properties
    SUB_DIRS = [x for x in path.iterdir() if x.is_dir()]
    configs = [read_config(sub_dir) for sub_dir in SUB_DIRS]

    BACKEND_NAME = configs[0].backend_name
    INSTANCE = configs[0].instance

    qubit_informations = [config.qubits for config in configs]
    # Flatten the list
    qubit_indices = [
        item.qubit_idx for sublist in qubit_informations for item in sublist
    ]

    service, backend_properties = get_backend_properies(
        BACKEND_NAME, INSTANCE, qubit_indices, status=status
    )

    # Read the executed data
    executed_dfs = [
        pd.read_csv(sub_dir / "executed_exp_data.csv")
        for sub_dir in path.iterdir()
        if sub_dir.is_dir()
    ]

    console.log("Waiting for the jobs to finish...")

    is_all_done = False
    while not is_all_done:
        if status is not None:
            status.update(BASE_THEME + "Checking the status of the jobs...")

        # Check the jobs status
        _jobs, is_all_done, is_error, num_done = sq.qiskit.check_jobs_status(
            executed_dfs[0], service=service, jobs=jobs
        )

        if is_error:
            # raise ValueError("Error in the job")
            # TODO: Add the error handling
            console.log("Error in the job")
            break

        if is_all_done:
            break
        else:
            if status is not None:
                status.update(
                    BASE_THEME
                    + "Waiting for the jobs to finish..."
                    + f"({num_done}/{len(_jobs)})"
                )

        time.sleep(wait_interval)

    console.log("The jobs are finished")

    if status is not None:
        status.update(BASE_THEME + "Retrieving the result from the backend...")

    # Filter only jobs that are done
    filtered_jobs = [
        job for job in _jobs if job.status() == "DONE" or job.status() == JobStatus.DONE
    ]
    # Filter dataframe with the job_id that are done
    filtered_dfs: list[pd.DataFrame] = []
    for df in executed_dfs:
        temp_df = df["job_id"].isin([job.job_id() for job in filtered_jobs])
        filtered_dfs.append(df[temp_df])

    # Retrieve the experiments
    result_dfs = [
        sq.qiskit.get_result_from_remote(
            executed_dfs[idx],
            filtered_jobs,
            partial(
                sq.qiskit.extract_results_v3, qubit_idx=qubit.qubit_idx, mcmd=False
            ),
        )
        for idx, qubit in enumerate(backend_properties.qubit_informations)
    ]

    if is_error:
        console.log("Error in the job, retrived only the done jobs")
        last_parameter_id = int(
            jnp.floor(
                (num_done * backend_properties.max_circuits_per_job)
                / len(configs[0].expectation_values_order)
            )
            - 1
        )

        # Filter the result_df up to the last parameter_id
        result_dfs = [df[df["parameters_id"] <= last_parameter_id] for df in result_dfs]

        # Change the number of sample_size in the configuration to the number of done jobs
        # log the message in the configuration.additional_info
        for config in configs:
            config.additional_info["original_sample_size"] = str(config.sample_size)
            config.sample_size = last_parameter_id + 1
            config.additional_info["message"] = (
                f"The sample size is changed to {config.sample_size} from {config.additional_info['original_sample_size']}"
            )
            console.log(config.additional_info["message"])

    # Try to save them
    for sub_dir, config, result_df in zip(SUB_DIRS, configs, result_dfs):
        exp_data = sq.data.ExperimentData(
            experiment_config=config, preprocess_data=result_df
        )
        # Save the experiment data
        exp_data.save_to_folder(path=sub_dir)

    console.log("The results are saved to the folder")


def step_04_train_model(
    path: Path,
    loss_choice: sq.optimizer.LossChoice,
    model_name: str = "",
    train_epochs: int = 1500,
    status: typing.Union[Status, None] = None,
) -> str:
    """
    Train the model using the experiment data
    and save the model to the folder
    """

    if status is not None:
        status.update(BASE_THEME + "Loading the data...")

    loaded_data = sq.utils.helper.load_data_from_path(path)

    # NOTE: flatten the pulse_parameters
    pulse_parameters = loaded_data.pulse_parameters.reshape(
        loaded_data.experiment_data.experiment_config.sample_size, -1
    )

    MASTER_KEY_SEED = loaded_data.experiment_data.experiment_config.additional_info[
        "MASTER_KEY_SEED"
    ]
    keys = sq.utils.helper.generate_keys(int(MASTER_KEY_SEED))

    if status is not None:
        status.update(BASE_THEME + "Training the model...")

    if loss_choice == sq.optimizer.LossChoice.MAEF:
        loss_fn = sq.model_v2.MAEF_loss_fn
        metric = "test/MAEF"
    else:
        loss_fn = sq.model_v2.MSEE_loss_fn
        metric = "test/MSEE"

    trainable = sq.optimizer.default_trainable_v2(
        pulse_sequence=loaded_data.pulse_sequence, loss_fn=loss_fn
    )

    results = sq.optimizer.hypertuner(
        trainable=trainable,
        pulse_parameters=jnp.array(pulse_parameters),
        unitaries=jnp.array(loaded_data.unitaries),
        expectations=jnp.array(loaded_data.expectation_values),
        model_key=keys.model_key,
        train_key=keys.random_split_key,
        num_samples=100,
        search_algo=sq.optimizer.SearchAlgo.OPTUNA,
        metric=metric,
    )

    model_state, history, _ = sq.optimizer.get_best_hypertuner_results(
        results, metric=metric
    )

    model = sq.model.BasicBlackBox(**model_state.model_config)
    model_params = model_state.model_params

    console.log("The model is trained")

    if status is not None:
        status.update(BASE_THEME + "Saving the model to the folder...")

    MODEL_PATH = path / "ckpts/" / model_name
    MODEL_PATH.mkdir(exist_ok=True, parents=True)

    history_df = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 1)
    ax = sns.lineplot(data=history_df, x="step", y=metric[5:], hue="loop")
    ax.set_yscale("log")
    fig.savefig(MODEL_PATH / "training_history.png")

    SAVED_MODEL_PATH = sq.model.save_model(
        str(MODEL_PATH),
        loaded_data.experiment_data.experiment_config.EXPERIMENT_IDENTIFIER,
        loaded_data.pulse_sequence,
        sq.utils.helper.rotating_transmon_hamiltonian,
        asdict(model),
        model_params,
        history,
        with_auto_datetime=False,
    )

    console.log("The model is saved to the folder")

    return SAVED_MODEL_PATH


def step_05_optimize_gate_using_model(
    path: Path,
    model_path: Path,
    status: typing.Union[Status, None] = None,
):
    """
    Optimize the gate using the trained model
    """

    if status is not None:
        status.update(BASE_THEME + "Loading the model...")

    model_state, restored_history, data_config = sq.model.load_model(str(model_path))

    loaded_data = sq.utils.helper.load_data_from_path(
        path,
    )

    console.log("The model is loaded")

    if status is not None:
        status.update(BASE_THEME + "Optimizing the gate...")

    exp_config = sq.data.ExperimentConfiguration.from_file(str(path))
    MASTER_KEY_SEED = exp_config.additional_info["MASTER_KEY_SEED"]
    TARGET_GATE = exp_config.additional_info["TARGET_GATE"]
    keys = sq.utils.helper.generate_keys(MASTER_KEY_SEED)

    model = sq.model.BasicBlackBox(**model_state.model_config)

    target_unitary = target_gate_choices[TARGET_GATE]
    parameter_structure = loaded_data.pulse_sequence.get_parameter_names()

    array_to_list_of_params_callable = lambda x: sq.pulse.array_to_list_of_params(
        x, parameter_structure
    )

    list_of_params_to_array_callable = lambda x: sq.pulse.list_of_params_to_array(
        x, parameter_structure
    )

    # using optax
    lower, upper = loaded_data.pulse_sequence.get_bounds()

    params = loaded_data.pulse_sequence.sample_params(keys.gate_optimization_key)

    lr_scheduler = optax.warmup_cosine_decay_schedule(
        init_value=1e-3,
        peak_value=1e-1,
        warmup_steps=100,
        decay_steps=1000,
        end_value=1e-3,
    )

    optimizer = optax.adam(lr_scheduler)

    cost = lambda pulse_param: sq.model_v2.gate_loss(
        list_of_params_to_array_callable(pulse_param),
        model,
        model_state.model_params,
        loaded_data.whitebox,
        array_to_list_of_params_callable,
        target_unitary,
    )

    # jit the function
    cost = jax.jit(cost)

    opt_pulse_params, losses, performances = sq.optimizer.optimize_v2(
        params, lower, upper, cost, optimizer, maxiter=1000
    )

    console.log("The gate is optimized")

    # Using jax.tree.map to convert the jax array to float
    opt_pulse_params = jax.tree.map(lambda x: float(x), opt_pulse_params)
    losses = jax.tree.map(lambda x: float(x), losses)
    performances = jax.tree.map(lambda x: float(x), performances)

    # Save the optimized pulse parameters to the folder with JSON format
    with open(model_path / "optimized_pulse_params.json", "w") as f:
        json.dump(opt_pulse_params, f)

    # Plot the optimized pulse parameters and save it to the folder
    fig, _ = loaded_data.pulse_sequence.draw(opt_pulse_params)
    fig.savefig(model_path / "optimized_pulse_params.png")

    console.log(performances[-1])

    # Save performance to the folder in JSON format
    with open(model_path / "performances.json", "w") as f:
        json.dump(performances, f)

    console.log("The performance is saved to the folder")


def step_05_2_optimize_pure_gate_with_physics(
    path: Path,
    model_path: Path,
    status: typing.Union[Status, None] = None,
):
    raise NotImplementedError("Not implemented yet")


def step_06_benchmark_gate(
    PATH: Path,
    MODEL_PATH: Path,
    NUM_REPEATS: int = 100,
    status: typing.Union[Status, None] = None,
):

    # Read the configuration from the folder
    config = sq.data.ExperimentConfiguration.from_file(str(PATH))
    # Read the pulse sequence from the folder
    pulse_sequence = sq.utils.helper.pulse_reader(str(PATH))
    # Get backend properties
    service, backend_properties = get_backend_properies(
        BACKEND_NAME=config.backend_name,
        INSTANCE=config.instance,
        qubit_indices=[config.qubits[0].qubit_idx],
    )

    if status is not None:
        status.update(BASE_THEME + "Loading the model...")

    # Read the optimized pulse parameters from the folder
    with open(MODEL_PATH / "optimized_pulse_params.json", "r") as f:
        opt_pulse_params = json.load(f)

    options = sq.qiskit.make_sampler_options(shots=config.shots)

    rows = []
    qcs = []
    for i in range(NUM_REPEATS):
        for idx, exp in enumerate(sq.utils.predefined.default_expectation_values_order):
            unique_id = (
                i * len(sq.utils.predefined.default_expectation_values_order) + idx
            )
            qc = sq.qiskit.create_circuit(
                initial_state=exp.initial_state,
                waveforms=[pulse_sequence.get_waveform(opt_pulse_params)],
                observable=exp.observable,
                qubit_indices=[config.qubits[0].qubit_idx],
                backend=backend_properties,
                add_pulse=not backend_properties.is_simulator,
                change_basis=True,
                add_measure=True,
                enable_MCMD=False,
                metadata={
                    "observable": exp.observable,
                    "initial_state": exp.initial_state,
                    "repeated_idx": i,
                    "circuit_idx": idx,
                    "unique_id": unique_id,
                },
            )

            rows.append(
                {
                    "observable": exp.observable,
                    "initial_state": exp.initial_state,
                    "repeated_idx": i,
                    "circuit_idx": idx,
                    "unique_id": unique_id,
                    "expectation_value": None,
                    "status": "created",
                    "submitted_at": None,
                }
            )

            qcs.append(qc)

    tqcs = sq.qiskit.transpile_circuits_v2(qcs, backend_properties, inital_layout=[config.qubits[0].qubit_idx])
    pre_execute_df = pd.DataFrame(rows)

    if status is not None:
        status.update(BASE_THEME + "Executing the circuits on the backend...")

    jobs = []
    rows = []

    with Session(
        service=service, backend=backend_properties.backend_instance
    ) as session:
        sampler = SamplerV2(mode=session, options=options)

        for i in range(NUM_REPEATS):

            start_idx = i * len(sq.utils.predefined.default_expectation_values_order)
            end_idx = (i + 1) * len(
                sq.utils.predefined.default_expectation_values_order
            )

            circuits_to_run = tqcs[start_idx:end_idx]

            job = sampler.run(circuits_to_run)
            # Job id
            job_id = job.job_id()
            # Save job id to the dataframe
            pre_execute_df.loc[pre_execute_df["repeated_idx"] == i, "job_id"] = job_id
            # Save submitted time to the dataframe
            pre_execute_df.loc[pre_execute_df["repeated_idx"] == i, "submitted_at"] = (
                pd.Timestamp.now()
            )

            jobs.append(job)

    if status is not None:
        status.update(BASE_THEME + "Saving the pre-execute dataframe to the folder...")

    BENCHMARK_PATH = (
        MODEL_PATH / "benchmark_results" / f"{sq.data.generate_path_with_datetime()}/"
    )
    BENCHMARK_PATH.mkdir(
        exist_ok=True, parents=True
    )  # NOTE: <--- create the folder with parent folders

    # Save the dataframe to the folder
    pre_execute_df.to_csv(BENCHMARK_PATH / "pre_execute_df.csv", index=False)

    return BENCHMARK_PATH, jobs


def step_07_retrieve_benchmark_results(
    PATH: Path,
    BENCMARK_PATH: Path,
    jobs: typing.Union[list[SamplerV2], None] = None,
    NUM_REPEATS: int = 100,
    status: typing.Union[Status, None] = None,
):
    config = sq.data.ExperimentConfiguration.from_file(str(PATH))
    target_unitary = np.array(
        target_gate_choices[config.additional_info["TARGET_GATE"]]
    )

    service, backend_properties = get_backend_properies(
        BACKEND_NAME=config.backend_name,
        INSTANCE=config.instance,
        qubit_indices=[config.qubits[0].qubit_idx],
    )

    pre_execute_df = pd.read_csv(BENCMARK_PATH / "pre_execute_df.csv")

    if jobs is None:
        jobs = []
        for job_id in pre_execute_df["job_id"].unique():
            job = service.job(job_id)
            jobs.append(job)

    extracted_func = lambda result, job_id: sq.qiskit.extract_results_v3(
        result=result, job_id=job_id, qubit_idx=config.qubits[0].qubit_idx, mcmd=False
    )

    if status is not None:
        status.update(BASE_THEME + "Retrieving the result from the backend...")

    result_df = sq.qiskit.get_result_from_remote(pre_execute_df, jobs, extracted_func)

    if status is not None:
        status.update(BASE_THEME + "Saving the result to the folder...")

    # Save the result to the folder
    result_df.to_csv(BENCMARK_PATH / "result_df.csv", index=False)

    rows = []

    if status is not None:
        status.update(BASE_THEME + "Calculating the average gate fidelity...")

    for i in range(NUM_REPEATS):

        result_df_slice = result_df[result_df["repeated_idx"] == i]

        formatted_expvals = []

        for row, exp in result_df_slice.iterrows():

            formatted_expvals.append(
                sq.data.ExpectationValue(
                    initial_state=exp["initial_state"],
                    observable=exp["observable"],
                    expectation_value=exp["expectation_value"],
                )
            )

        temp_expvals: list[float] = []
        for idx, exp in enumerate(sq.constant.default_expectation_values_order):
            temp_expvals.append(
                result_df_slice[
                    (result_df_slice["observable"] == exp.observable)
                    & (result_df_slice["initial_state"] == exp.initial_state)
                ]["expectation_value"].values[0]
            )

        reconstructed_superop = sq.physics.choi2superop(
            sq.physics.forest_process_tomography(formatted_expvals)
        )
        avg_gate_fidelity = sq.physics.avg_gate_fidelity_from_superop(
            reconstructed_superop, sq.physics.to_superop(jnp.array(target_unitary))
        )

        coefficients = sq.model.direct_AFG_estimation_coefficients(
            jnp.array(target_unitary)
        )
        direct_avg_gate_fidelity = sq.model.direct_AFG_estimation(
            coefficients, jnp.array(temp_expvals)
        )

        rows.append(
            {
                "repeated_idx": i,
                "avg_gate_fidelity": float(avg_gate_fidelity),
                "direct_avg_gate_fidelity": direct_avg_gate_fidelity,
                "submitted_at": result_df_slice["submitted_at"].min(),
            }
        )

    if status is not None:
        status.update(BASE_THEME + "Saving the average gate fidelity to the folder...")

    avg_gate_fidelity_df = pd.DataFrame(rows)
    avg_gate_fidelity_df.to_csv(BENCMARK_PATH / "avg_gate_fidelity_df.csv", index=False)


app = typer.Typer(no_args_is_help=True)


@app.command()
def prepare_parallel_experiment(path: str):

    PATH = Path(path)
    SUB_DIRS = [x for x in PATH.iterdir() if x.is_dir()]

    with console.status(BASE_THEME + "Setting up the experiment...") as status:
        # Read the configuration files within the directory
        configs = [read_config(sub_dir) for sub_dir in SUB_DIRS]

        # Ensure that the backend for each sub-experiment is the same
        BACKEND_NAME = configs[0].backend_name
        INSTANCE = configs[0].instance
        qubit_indice = [config.qubits for config in configs]
        # Flatten the list
        qubit_indices = [item.qubit_idx for sublist in qubit_indice for item in sublist]

        for config in configs:
            if config.backend_name != BACKEND_NAME:
                raise ValueError("The backend for each sub-experiment must be the same")
            if config.instance != INSTANCE:
                raise ValueError(
                    "The instance for each sub-experiment must be the same"
                )

        service, backend_properties = get_backend_properies(
            BACKEND_NAME, INSTANCE, qubit_indices, status
        )

        step_01_prepare_parallel_experiment(SUB_DIRS, backend_properties, status)


@app.command()
def execute_parallel_experiment(path: str, wait: bool = False, wait_interval: int = 10):

    PATH = Path(path)
    SUB_DIRS = [x for x in PATH.iterdir() if x.is_dir()]

    with console.status(BASE_THEME + "Setting up the experiment...") as status:

        jobs = step_02_execute_parallel_experiment(SUB_DIRS, status)

        if wait:
            step_03_wait_for_jobs(
                PATH, jobs=jobs, wait_interval=wait_interval, status=status
            )


@app.command()
def wait_for_jobs(path: str, wait_interval: int = 10):

    PATH = Path(path)

    with console.status(BASE_THEME + "Waiting for the jobs to finish...") as status:
        step_03_wait_for_jobs(PATH, wait_interval=wait_interval, status=status)


@app.command()
def train_model(path: str, loss: sq.optimizer.LossChoice):

    PATH = Path(path)
    SUB_DIRS = [x for x in PATH.iterdir() if x.is_dir()]

    with console.status(BASE_THEME + "Training the model...") as status:

        for sub_dir in SUB_DIRS:

            model_path = step_04_train_model(sub_dir, loss_choice=loss, model_name=loss.value, status=status)
            step_05_optimize_gate_using_model(sub_dir, Path(model_path), status=status)


@app.command()
def optimize_gate(path: str, model_name: str = "", pure_gate: bool = False):

    PATH = Path(path)
    SUB_DIRS = [x for x in PATH.iterdir() if x.is_dir()]

    with console.status(BASE_THEME + "Training the model...") as status:

        for sub_dir in SUB_DIRS:
            model_path = sub_dir / "ckpts" / model_name
            if pure_gate:
                step_05_2_optimize_pure_gate_with_physics(
                    sub_dir, Path(model_path), status=status
                )
            else:
                step_05_optimize_gate_using_model(
                    sub_dir, Path(model_path), status=status
                )


@app.command()
def batch_benchmark_gate(
    folder_path: str,
    model_path: str,
    num_repeats: int = 100,
    wait: bool = False,
):

    PATH = Path(folder_path)
    MODEL_PATH = Path(model_path)

    with console.status(BASE_THEME + "Benchmarking gate...") as status:

        BENCHMARK_PATH, jobs = step_06_benchmark_gate(
            PATH, MODEL_PATH, num_repeats, status
        )

        if wait:
            step_07_retrieve_benchmark_results(
                PATH, BENCHMARK_PATH, jobs, num_repeats, status
            )


@app.command()
def retrive_benchmark_gate(
    folder_path: str,
    model_path: str,
    benchmark_path: str,
    num_repeats: int = 100,
):

    PATH = Path(folder_path)
    MODEL_PATH = Path(model_path)
    BENCHMARK_PATH = Path(benchmark_path)

    with console.status(BASE_THEME + "Benchmarking gate...") as status:

        step_07_retrieve_benchmark_results(
            PATH, BENCHMARK_PATH, None, num_repeats, status
        )


if __name__ == "__main__":
    app()
