from pathlib import Path
import typing
import inspeqtor as sq  # type: ignore
from rich.console import Console
from rich.status import Status

# For testing
from qiskit_ibm_runtime.fake_provider import FakeJakartaV2  # type: ignore

console = Console()
BASE_THEME = "[bold green]:cold_face: "
target_gate_choices = {
    "X": sq.constant.X,
    "Y": sq.constant.Y,
    "Z": sq.constant.Z,
    "SX_v2": sq.constant.SX,
}


def read_config(
    path: Path,
) -> sq.data.ExperimentConfiguration:
    """Read the configuration from given path

    Args:
        path (Path): Path object to read configuration

    Raises:
        FileNotFoundError: Configuration file does not exist
        FileNotFoundError: Pulse sequence config file does not exist

    Returns:
        sq.data.ExperimentConfiguration: Experiment Configuration object
    """

    # Check that configuration file exists
    if not (path / "config.json").exists():
        raise FileNotFoundError(
            f"Configuration file does not exist in the folder: {path}"
        )

    # Check that pulse sequence file exists
    if not (path / "pulse_sequence.json").exists():
        raise FileNotFoundError(
            f"Pulse sequence file does not exist in the folder: {path}"
        )

    # read the configuration from the folder
    config = sq.data.ExperimentConfiguration.from_file(str(path))

    return config


def get_backend_properies(
    BACKEND_NAME: str,
    INSTANCE: str,
    qubit_indices: list[int],
    status: typing.Union[Status, None] = None,
):

    is_simulator = True if "fake" in BACKEND_NAME else False

    if status is not None:
        status.update(BASE_THEME + "Retrieving the backend information...")

    if not is_simulator:
        console.log(f"Using real backend: {BACKEND_NAME}")
        service, backend = sq.qiskit.get_ibm_service_and_backend(
            instance=INSTANCE, backend_name=BACKEND_NAME
        )
    else:
        console.log("Using fake backend")
        backend = FakeJakartaV2()
        service = None

    backend_properties = sq.qiskit.IBMQDeviceProperties.from_backend(
        backend=backend, qubit_indices=qubit_indices, service=service
    )

    return service, backend_properties


def rewrite_qubit_information(
    path: Path,
    qubit_informations: list[sq.data.QubitInformation],
):

    config = sq.data.ExperimentConfiguration.from_file(str(path))
    config.qubits = qubit_informations
    config.to_file(str(path))

    return config


def gen_command(
    path,
    with_prepare=True,
    with_execute=True,
    with_wait=True,
    with_train=True,
    with_benchmark=True,
):

    prepare_command = f"python receipt.py prepare-parallel-experiment {str(path)}/"
    execute_command = f"python receipt.py execute-parallel-experiment {str(path)}/"
    wait_for_job_command = (
        f"python receipt.py wait-for-jobs {str(path)}/ --wait-interval=600"
    )
    train_model_command = f"python receipt.py train-model {str(path)}/"
    benchmark_command = (
        f"python receipt.py batch-benchmark-gate {str(path)}/q0/ {str(path)}/q0/ckpts/"
    )

    if with_wait and with_execute:
        execute_command += " --wait"
        with_wait = False

    commands = []
    if with_prepare:
        commands.append(prepare_command)
    if with_execute:
        commands.append(execute_command)
    if with_wait:
        commands.append(wait_for_job_command)
    if with_train:
        commands.append(train_model_command)
    if with_benchmark:
        commands.append(benchmark_command)

    # Join commands with " && "
    return " && ".join(commands)


def submit_experiment_command(path):
    return gen_command(
        path,
        with_prepare=True,
        with_execute=True,
        with_benchmark=False,
        with_wait=False,
        with_train=False,
    )


def analyze_experiment_command(path):
    return gen_command(
        path,
        with_prepare=False,
        with_execute=False,
        with_benchmark=True,
        with_wait=True,
        with_train=True,
    )


def test_receipt_command(path):
    return gen_command(
        path,
        with_prepare=True,
        with_execute=True,
        with_wait=True,
        with_train=True,
        with_benchmark=False,
    )
