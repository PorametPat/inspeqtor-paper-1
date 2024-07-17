import jax
import jax.numpy as jnp
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, pulse, qpy  # type: ignore
from qiskit.circuit import Gate  # type: ignore
from qiskit.providers import BackendV2, JobStatus  # type: ignore
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager  # type: ignore
from dataclasses import dataclass
from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2  # type: ignore
from qiskit.providers.models import BackendProperties  # type: ignore
from qiskit_experiments.framework import BackendData  # type: ignore
import pandas as pd  # type: ignore
from typing import Any, Union, Callable
from alive_progress import alive_it  # type: ignore
from qiskit.primitives import PrimitiveJob, SamplerResult, PrimitiveResult  # type: ignore
from qiskit_ibm_runtime import RuntimeJobV2, RuntimeJob, SamplerV2, QiskitRuntimeService, Session  # type: ignore
from qiskit.providers import JobStatus
from qiskit_ibm_runtime.options import DynamicalDecouplingOptions, TwirlingOptions, SamplerOptions  # type: ignore
from typing import Sequence
from dataclasses import dataclass

from .data import (
    QubitInformation,
    ExperimentConfiguration,
    make_row,
)
from .typing import ParametersDictType
from .pulse import JaxBasedPulseSequence


@dataclass
class CompackConfig:
    identifier: str
    shots: int
    sample_size: int
    qubit_idx: int


def get_qubit_information_from_backend(
    qubit_idx: int,
    device_configuration: dict,
    device_properties: dict,
) -> QubitInformation:

    backend_properties = BackendProperties.from_dict(device_properties)

    anham, _ = backend_properties.qubit_property(qubit_idx, "anharmonicity")
    freq, _ = backend_properties.qubit_property(qubit_idx, "frequency")

    assert isinstance(anham, float) and isinstance(freq, float)

    anham = anham / (1e9)
    freq = freq / (1e9)

    drive_str = device_configuration["hamiltonian"]["vars"][f"omegad{qubit_idx}"] / (
        2 * np.pi
    )

    return QubitInformation(
        unit="GHz",
        qubit_idx=qubit_idx,
        anharmonicity=anham,
        frequency=freq,
        drive_strength=drive_str,
    )


@dataclass
class IBMQDeviceProperties:
    """
    qubit_informations: A list of QubitInformationV3 objects representing
    granularity: An integer value representing minimum pulse gate
        resolution in units of dt. A user-defined pulse gate should have duration of a multiple of this granularity value.
    min_length: An integer value representing minimum pulse gate
        length in units of dt. A user-defined pulse gate should be longer than this length.
    pulse_alignment: An integer value representing a time resolution of gate
        instruction starting time. Gate instruction should start at time which is a multiple of the alignment value.
    acquire_alignment: An integer value representing a time resolution of measure
        instruction starting time. Measure instruction should start at time which is a multiple of the alignment value.
    """

    qubit_informations: list[QubitInformation]
    acquire_alignment: int
    granularity: int
    min_length: int
    pulse_alignment: int
    dt: float
    backend_instance: BackendV2
    max_circuits_per_job: int
    name: str
    is_simulator: bool
    hamiltonian_data: dict[str, Any]

    @classmethod
    def from_backend(
        cls,
        backend: BackendV2,
        qubit_indices: list[int] = [0],
        service: QiskitRuntimeService | None = None,
    ) -> "IBMQDeviceProperties":

        if isinstance(backend, FakeBackendV2):
            # It is necessary to set the properties dict from the json file
            backend._set_props_dict_from_json()
            device_configuration = backend._get_conf_dict_from_json()
            device_properties = backend._props_dict

        else:

            if service is None:
                raise ValueError(
                    "service must be provided when backend is not a FakeBackendV2 instance"
                )
            # Get the backend configuration and properties from the service
            device_configuration = service._api_client.backend_configuration(
                backend.name
            )
            device_properties = service._api_client.backend_properties(backend.name)

        assert isinstance(device_configuration, dict) and isinstance(
            device_properties, dict
        )

        hamiltonian_data = device_configuration["hamiltonian"]

        qubit_informations = [
            get_qubit_information_from_backend(
                q_idx, device_configuration, device_properties
            )
            for q_idx in qubit_indices
        ]

        backend_data = BackendData(backend)

        dt = backend.dt
        assert isinstance(dt, float)
        dt = dt * 1e9  # Convert to ns
        acquire_alignment = backend_data.acquire_alignment
        granularity = backend_data.granularity
        min_length = backend_data.min_length
        pulse_alignment = backend_data.pulse_alignment
        max_circuits = backend_data.max_circuits if backend_data.max_circuits else 75
        name = backend.name
        is_simulator = True if "fake" in backend.name else False

        return cls(
            qubit_informations=qubit_informations,
            dt=dt,
            backend_instance=backend,
            max_circuits_per_job=max_circuits,
            acquire_alignment=acquire_alignment,
            granularity=granularity,
            min_length=min_length,
            pulse_alignment=pulse_alignment,
            name=name,
            is_simulator=is_simulator,
            hamiltonian_data=hamiltonian_data,
        )


def get_circuit(
    initial_state: str,
    waveforms: np.ndarray,
    observable: str,
    q_idx: int,
    backend: IBMQDeviceProperties,
    add_pulse: bool = True,
    change_basis: bool = False,
    add_measure: bool = False,
    enable_MCMD: bool = False,
    metadata: dict = {},
) -> QuantumCircuit:
    """
    Create a quantum circuit based on the given inputs.

    Args:
        initial_state (str): The initial state of the qubit. Can be '1', '+', '-', 'r', or 'l'.
        waveforms (np.ndarray): The waveform to be applied to the qubit.
        observable (str): The observable to be measured.
        q_idx (int): The index of the qubit.
        backend (BackendProperties): The backend to be used for building the circuit.
        add_pulse (bool, optional): Whether to add a pulse to the circuit. Defaults to True.
        change_basis (bool, optional): Whether to apply a basis change to the circuit. Defaults to False.
        add_measure (bool, optional): Whether to add a measurement to the circuit. Defaults to False.
        enable_MCMD (bool, optional): Whether to enable multi-channel measurement discrimination. Defaults to False.
        metadata (dict, optional): Additional metadata for the circuit. Defaults to an empty dictionary.

    Returns:
        QuantumCircuit: The constructed quantum circuit based on the given inputs.
    """

    if add_measure:
        qc = QuantumCircuit(1, 1, metadata=metadata)
    if enable_MCMD:
        qc = QuantumCircuit(1, 2, metadata=metadata)
    else:
        qc = QuantumCircuit(1)

    if initial_state == "1":
        qc.x(q_idx)
    elif initial_state == "+":
        qc.h(q_idx)
    elif initial_state == "-":
        qc.x(q_idx)
        qc.h(q_idx)
    elif initial_state == "r":
        qc.h(q_idx)
        qc.s(q_idx)
    elif initial_state == "l":
        qc.h(q_idx)
        qc.sdg(q_idx)

    if add_pulse:
        with pulse.build(backend=backend.backend_instance, name="test") as test_pulse:

            # Check if the waveform is shorter than the minimum length
            if waveforms.shape[0] < backend.min_length:
                # Zero pad the waveform
                waveforms = np.pad(
                    waveforms,
                    (0, backend.min_length - waveforms.shape[0]),
                    mode="constant",
                )

            drive_channel = pulse.drive_channel(q_idx)
            pulse.play(pulse.Waveform(waveforms), drive_channel)

        qc.add_calibration("pulse_sequence", [q_idx], test_pulse)
        custom_gate = Gate(name="pulse_sequence", num_qubits=1, params=[])
        qc.append(custom_gate, [q_idx])

    if change_basis:
        if observable == "X":
            qc.h(q_idx)
        elif observable == "Y":
            qc.sdg(q_idx)
            qc.h(q_idx)

    if add_measure:
        qc.measure(q_idx, 0)

    if enable_MCMD:
        qc.x(q_idx)
        qc.measure(q_idx, 1)

    return qc


def get_ibm_service_and_backend(
    instance: str, backend_name: str
) -> tuple[QiskitRuntimeService, BackendV2]:

    service = QiskitRuntimeService(instance=instance)
    backend = service.backend(backend_name)

    return service, backend


def prepare_experiment(
    config: ExperimentConfiguration,
    pulse_sequence: JaxBasedPulseSequence,
    key: jnp.ndarray,
    backend_properties: IBMQDeviceProperties,
) -> tuple[pd.DataFrame, list[QuantumCircuit]]:

    rows = []
    circuit_idx = 0
    job_idx = 0
    unique_id = 0
    quantum_circuits = []

    IS_REAL_BACKEND = not "fake" in backend_properties.name

    for params_id in range(config.sample_size):

        # Split the pulse_key
        key, sample_key = jax.random.split(key)
        # Get the parameters for this sample
        params = pulse_sequence.sample_params(sample_key)
        waveform = pulse_sequence.get_waveform(params)

        for expectation_value in config.expectation_values_order:
            row = make_row(
                expectation_value=jnp.nan,
                initial_state=expectation_value.initial_state,
                observable=expectation_value.observable,
                parameters_id=params_id,
                parameters_list=params,
                job_id="NOT_YET_ASSIGNED",
                job_idx=job_idx,
                unique_id=unique_id,
                circuit_idx=circuit_idx,
            )

            rows.append(row)

            circuit = get_circuit(
                initial_state=expectation_value.initial_state,
                waveforms=np.array(waveform),
                observable=expectation_value.observable,
                q_idx=config.qubits[0].qubit_idx,
                backend=backend_properties,
                add_pulse=IS_REAL_BACKEND,
                change_basis=True,
                add_measure=True,
                enable_MCMD=True,
                metadata={
                    "unqiue_id": unique_id,
                },
            )

            quantum_circuits.append(circuit)

            # increment unique_id
            unique_id += 1
            # increment circuit_index
            circuit_idx += 1
            # if the circuit_index is greater than the number of circuits per job
            if circuit_idx == backend_properties.max_circuits_per_job:
                # reset circuit_index
                circuit_idx = 0
                # increment job_index
                job_idx += 1

    return pd.DataFrame(rows), quantum_circuits


def assert_list_of_quantum_circuits(quantum_circuits: Any) -> list[QuantumCircuit]:

    temp_list: list[QuantumCircuit] = []
    for qc in quantum_circuits:
        assert isinstance(qc, QuantumCircuit)
        temp_list.append(qc)

    return temp_list


def transpile_circuits(
    quantum_circuits: list[QuantumCircuit],
    backend: BackendV2,
    inital_layout: list[int],
    num_processes: int = 1,
) -> list[QuantumCircuit]:

    pass_manager = generate_preset_pass_manager(
        optimization_level=0,
        backend=backend,
        initial_layout=inital_layout,
    )

    transpiled_qc_list = pass_manager.run(quantum_circuits, num_processes=num_processes)

    transpiled_qc_list = assert_list_of_quantum_circuits(transpiled_qc_list)
    return transpiled_qc_list


def save_quantum_circuits(quantum_circuits: list[QuantumCircuit], file_name: str):

    # Save the transpiled quantum circuits
    with open(file_name, "wb") as fd:
        qpy.dump(quantum_circuits, fd)  # type: ignore


def load_quantum_circuits(file_name: str) -> list[QuantumCircuit]:

    with open(file_name, "rb") as fd:
        quantum_circuits = qpy.load(fd)

    quantum_circuits = assert_list_of_quantum_circuits(quantum_circuits)

    return quantum_circuits


def execute_experiment(
    dataframe: pd.DataFrame,
    backend_properties: IBMQDeviceProperties,
    options,
    transpiled_circuits: list[QuantumCircuit],
    service: QiskitRuntimeService | None,
    Sampler: SamplerV2,
) -> tuple[pd.DataFrame, list[PrimitiveJob]]:

    total_circuits = dataframe.shape[0]
    total_jobs = total_circuits // backend_properties.max_circuits_per_job
    if total_circuits % backend_properties.max_circuits_per_job != 0:
        total_jobs += 1

    jobs = []
    with Session(
        service=service, backend=backend_properties.backend_instance
    ) as session:
        sampler = Sampler(mode=session, options=options)

        for job_idx in alive_it(range(total_jobs)):
            # Get the start and end index for the circuits
            start_idx = int(job_idx * backend_properties.max_circuits_per_job)
            end_idx = int((job_idx + 1) * backend_properties.max_circuits_per_job)
            # Get a portion of the circuits
            circuits_to_run = transpiled_circuits[start_idx:end_idx]
            # Run the circuits
            job = sampler.run(circuits_to_run)
            # Get the job_id
            job_id = job.job_id()
            # Save the job_id to the dataframe
            dataframe.loc[start_idx:end_idx, "job_id"] = job_id
            # Append the job to the jobs list
            jobs.append(job)

    return dataframe, jobs


def extract_results_v1(
    result: SamplerResult,
    job_id: str,
) -> pd.DataFrame:

    raw_rows = []
    for circuit_idx, (metadata, quasi_dist) in enumerate(
        zip(result.metadata, result.quasi_dists)
    ):
        quasi_dist_0 = quasi_dist.get(0, 0)  # |unknown>
        quasi_dist_1 = quasi_dist.get(1, 0)  # |1>
        quasi_dist_2 = quasi_dist.get(2, 0)  # |0>
        quasi_dist_3 = quasi_dist.get(3, 0)  # |leakage>

        expval = (quasi_dist_2 - quasi_dist_1) / (quasi_dist_2 + quasi_dist_1)

        raw_rows.append(
            {
                "job_id": job_id,
                "unique_id": metadata.get("circuit_metadata", {}).get("unique_id", -1),
                "circuit_idx": circuit_idx,
                "expectation_value": expval,
                "counts/00": quasi_dist_0,
                "counts/01": quasi_dist_1,
                "counts/10": quasi_dist_2,
                "counts/11": quasi_dist_3,
                "status": "COMPLETED",
            }
        )

    return pd.DataFrame(raw_rows)


def extract_results_v2(result, job_id: str) -> pd.DataFrame:

    raw_rows = []

    for circuit_idx, res in enumerate(result):
        # Get the counts
        counts = res.data.c.get_counts()
        num_shots = (
            res.data.c.num_shots
        )  # NOTE: .c because of the name of classical register https://docs.quantum.ibm.com/api/migration-guides/v2-primitives#steps-to-migrate-to-sampler-v2
        assert isinstance(counts, dict) and isinstance(num_shots, int)

        c_r00 = counts.get("00", 0)  # |unknown>
        c_r01 = counts.get("01", 0)  # |1>
        c_r10 = counts.get("10", 0)  # |0>
        c_r11 = counts.get("11", 0)  # |leakage>

        # Calculate the expectation values
        expval = (c_r10 - c_r01) / (c_r10 + c_r01)

        raw_rows.append(
            {
                "job_id": job_id,
                "circuit_idx": circuit_idx,
                "unique_id": res.metadata.get("circuit_metadata", {}).get(
                    "unique_id", -1
                ),
                "expectation_value": expval,
                "counts/00": c_r00,
                "counts/01": c_r01,
                "counts/10": c_r10,
                "counts/11": c_r11,
                "status": "COMPLETED",
            }
        )

    return pd.DataFrame(raw_rows)


def get_result_from_remote(
    execute_dataframe: pd.DataFrame,
    jobs: list[PrimitiveJob],
    extract_function: Callable[[Any, str], pd.DataFrame],
):

    dataframe = execute_dataframe.copy()
    dataframe.set_index(["circuit_idx", "job_id"], inplace=True)

    # Add "counts/**" columns to the dataframe
    for bitstring in ["00", "01", "10", "11"]:
        dataframe[f"counts/{bitstring}"] = np.nan

    for job in jobs:

        result = job.result()
        raw_df = extract_function(result, job.job_id())
        raw_df.set_index(["circuit_idx", "job_id"], inplace=True)
        dataframe.update(raw_df)

        if isinstance(job, RuntimeJobV2):
            # Update timestampes
            timestamps_names = ["created", "running", "finished"]
            for name in timestamps_names:
                dataframe.loc[(slice(None), job.job_id()), (name)] = job.metrics()[
                    "timestamps"
                ][name]

    return dataframe.reset_index()


def check_jobs_status(
    dataframe: pd.DataFrame,
    service: QiskitRuntimeService | None = None,
    jobs: list[PrimitiveJob] | None = None,
) -> tuple[list[PrimitiveJob | RuntimeJob | RuntimeJobV2], bool, bool, int]:

    job_ids = dataframe["job_id"].unique()

    statuses = []
    _jobs = []

    if service is not None:
        for job_id in job_ids:

            job = service.job(job_id)
            statuses.append(job.status())
            _jobs.append(job)
    elif jobs is not None:
        for job in jobs:
            statuses.append(job.status())
            _jobs.append(job)
    else:
        raise ValueError("Either jobs or service must be provided")

    # Check the status of the jobs
    is_all_done = all(
        [(status == JobStatus.DONE) or (status == "DONE") for status in statuses]
    )

    # Check if some of the jobs are error or cancelled
    is_error = any(
        [
            status == JobStatus.ERROR
            or status == JobStatus.CANCELLED
            or status == "ERROR"
            or status == "CANCELLED"
            for status in statuses
        ]
    )

    # Count the number of jobs that are done
    num_done = sum(
        [(status == JobStatus.DONE) or (status == "DONE") for status in statuses]
    )

    return _jobs, is_all_done, is_error, num_done


def transpile_circuits_v2(
    quantum_circuits: list[QuantumCircuit],
    backend_properties: IBMQDeviceProperties,
    inital_layout: list[int],
    num_processes: int = 1,
) -> list[QuantumCircuit]:

    pass_manager = generate_preset_pass_manager(
        optimization_level=0,
        backend=backend_properties.backend_instance,
        initial_layout=inital_layout,
        scheduling_method="asap",
        timing_constraints=backend_properties,
        layout_method="trivial",
    )

    transpiled_qc_list = pass_manager.run(quantum_circuits, num_processes=num_processes)

    transpiled_qc_list = assert_list_of_quantum_circuits(transpiled_qc_list)
    return transpiled_qc_list


from qiskit import QuantumRegister, ClassicalRegister


def create_circuit(
    initial_state: str,
    waveforms: list[np.ndarray],
    observable: str,
    qubit_indices: list[int],
    backend: IBMQDeviceProperties,
    add_pulse: bool = True,
    change_basis: bool = False,
    add_measure: bool = False,
    enable_MCMD: bool = False,
    metadata: dict = {},
) -> QuantumCircuit:
    """
    Create a quantum circuit based on the given inputs.

    Args:
        initial_state (str): The initial state of the qubit. Can be '1', '+', '-', 'r', or 'l'.
        waveforms (np.ndarray): The waveform to be applied to the qubit.
        observable (str): The observable to be measured.
        q_idx (int): The index of the qubit.
        backend (BackendProperties): The backend to be used for building the circuit.
        add_pulse (bool, optional): Whether to add a pulse to the circuit. Defaults to True.
        change_basis (bool, optional): Whether to apply a basis change to the circuit. Defaults to False.
        add_measure (bool, optional): Whether to add a measurement to the circuit. Defaults to False.
        enable_MCMD (bool, optional): Whether to enable multi-channel measurement discrimination. Defaults to False.
        metadata (dict, optional): Additional metadata for the circuit. Defaults to an empty dictionary.

    Returns:
        QuantumCircuit: The constructed quantum circuit based on the given inputs.
    """
    num_qubits = len(qubit_indices)

    qrs = QuantumRegister(num_qubits)
    crs = [ClassicalRegister(2, name=f"c{qidx}") for qidx in qubit_indices]

    if add_measure:
        qc = QuantumCircuit(qrs, *crs, metadata=metadata)
    if enable_MCMD:
        qc = QuantumCircuit(qrs, *crs, metadata=metadata)
    else:
        qc = QuantumCircuit(qrs)

    if initial_state == "1":
        qc.x(qrs)
    elif initial_state == "+":
        qc.h(qrs)
    elif initial_state == "-":
        qc.x(qrs)
        qc.h(qrs)
    elif initial_state == "r":
        qc.h(qrs)
        qc.s(qrs)
    elif initial_state == "l":
        qc.h(qrs)
        qc.sdg(qrs)

    if add_pulse:
        with pulse.build(backend=backend.backend_instance, name=f"pulse") as test_pulse:

            for q_idx, waveform in zip(qubit_indices, waveforms):
                # Check if the waveform is shorter than the minimum length
                if waveform.shape[0] < backend.min_length:
                    # Zero pad the waveform
                    waveform = np.pad(
                        waveform,
                        (0, backend.min_length - waveform.shape[0]),
                        mode="constant",
                    )

                drive_channel = pulse.drive_channel(q_idx)
                pulse.play(pulse.Waveform(waveform), drive_channel)

        qc.add_calibration(f"pulse", qubit_indices, test_pulse)
        custom_gate = Gate(name=f"pulse", num_qubits=num_qubits, params=[])
        # qc.append(custom_gate, qrs)
        qc.append(custom_gate, list(range(num_qubits)))

    if change_basis:
        if observable == "X":
            qc.h(qrs)
        elif observable == "Y":
            qc.sdg(qrs)
            qc.h(qrs)

    if add_measure:
        for i in range(num_qubits):
            qc.measure(qrs[i], crs[i][0])

    if enable_MCMD:
        qc.x(qrs)
        for i in range(num_qubits):
            qc.measure(qrs[i], crs[i][1])

    return qc


@dataclass
class QuantumCircuitData:
    metadata: dict
    parameter: list[ParametersDictType]
    initial_state: str
    observable: str
    qubit_idx: int


def prepare_experiment_v2(
    config: ExperimentConfiguration,
    pulse_sequence: JaxBasedPulseSequence,
    key: jnp.ndarray,
    backend_properties: IBMQDeviceProperties,
) -> tuple[pd.DataFrame, list[QuantumCircuitData]]:

    rows = []
    circuit_idx = 0
    job_idx = 0
    unique_id = 0
    quantum_circuits = []

    for params_id in range(config.sample_size):

        # Split the pulse_key
        key, sample_key = jax.random.split(key)
        # Get the parameters for this sample
        params = pulse_sequence.sample_params(sample_key)

        for expectation_value in config.expectation_values_order:
            row = make_row(
                expectation_value=jnp.nan,
                initial_state=expectation_value.initial_state,
                observable=expectation_value.observable,
                parameters_id=params_id,
                parameters_list=params,
                job_id="NOT_YET_ASSIGNED",
                job_idx=job_idx,
                unique_id=unique_id,
                circuit_idx=circuit_idx,
            )

            rows.append(row)

            quantum_circuits.append(
                QuantumCircuitData(
                    metadata={
                        "unqiue_id": unique_id,
                    },
                    parameter=params,
                    initial_state=expectation_value.initial_state,
                    observable=expectation_value.observable,
                    qubit_idx=config.qubits[0].qubit_idx,
                )
            )

            # increment unique_id
            unique_id += 1
            # increment circuit_index
            circuit_idx += 1
            # if the circuit_index is greater than the number of circuits per job
            if circuit_idx == backend_properties.max_circuits_per_job:
                # reset circuit_index
                circuit_idx = 0
                # increment job_index
                job_idx += 1

    return pd.DataFrame(rows), quantum_circuits


def prepare_parallel_experiment_v2(
    configs: list[ExperimentConfiguration],
    pulse_sequences: list[JaxBasedPulseSequence],
    keys: Sequence[jnp.ndarray],
    backend_properties: IBMQDeviceProperties,
) -> tuple[list[pd.DataFrame], list[QuantumCircuit]]:

    dataframes: list[pd.DataFrame] = []
    quantum_circuits: list[list[QuantumCircuitData]] = []

    for i, (conf, pulse_seq, key) in enumerate(zip(configs, pulse_sequences, keys)):
        dataframe, qcs = prepare_experiment_v2(
            config=conf,
            pulse_sequence=pulse_seq,
            key=key,
            backend_properties=backend_properties,
        )

        assert len(dataframe) == len(
            qcs
        ), "The len of dataframe and Quantum circuit is not the same"

        dataframes.append(dataframe)
        quantum_circuits.append(qcs)

    # Create the quantum circuits
    combined_qcs = []
    for qcs in zip(*quantum_circuits):  # type: ignore

        # Combind the metadata
        metadata = {}
        for qc in qcs:
            assert isinstance(qc, QuantumCircuitData)
            metadata[qc.qubit_idx] = qc.metadata

        combined_qc = create_circuit(
            initial_state=qcs[0].initial_state,
            waveforms=[
                np.array(pulse_seq.get_waveform(qc.parameter))
                for pulse_seq, qc in zip(pulse_sequences, qcs)
            ],
            observable=qcs[0].observable,
            qubit_indices=[qc.qubit_idx for qc in qcs],
            backend=backend_properties,
            add_pulse=not backend_properties.is_simulator,
            change_basis=True,
            add_measure=True,
            enable_MCMD=True,
            metadata=metadata,
        )
        combined_qcs.append(combined_qc)

    assert len(combined_qcs) == len(dataframes[0])

    return dataframes, combined_qcs


def retrieve_keys(MASTER_KEY_SEED: int):
    master_key = jax.random.PRNGKey(MASTER_KEY_SEED)
    pulse_random_split_key, random_split_key, model_key, dropout_key, gate_optim_key = (
        jax.random.split(master_key, 5)
    )
    return (
        pulse_random_split_key,
        random_split_key,
        model_key,
        dropout_key,
        gate_optim_key,
    )


def execute_parallel_experiment(
    dataframes: list[pd.DataFrame],
    backend_properties: IBMQDeviceProperties,
    options: SamplerOptions,
    transpiled_circuits: list[QuantumCircuit],
    service: QiskitRuntimeService | None,
) -> tuple[list[pd.DataFrame], list[PrimitiveJob]]:

    total_circuits = len(transpiled_circuits)
    total_jobs = total_circuits // backend_properties.max_circuits_per_job
    if total_circuits % backend_properties.max_circuits_per_job != 0:
        total_jobs += 1

    jobs = []
    with Session(
        service=service, backend=backend_properties.backend_instance
    ) as session:
        sampler = SamplerV2(mode=session, options=options, session=session)

        for job_idx in alive_it(range(total_jobs)):
            # Get the start and end index for the circuits
            start_idx = int(job_idx * backend_properties.max_circuits_per_job)
            end_idx = int((job_idx + 1) * backend_properties.max_circuits_per_job)
            # Get a portion of the circuits
            circuits_to_run = transpiled_circuits[start_idx:end_idx]
            # Run the circuits
            job = sampler.run(circuits_to_run)
            # Get the job_id
            job_id = job.job_id()
            # Save the job_id to the dataframe
            for dataframe in dataframes:
                dataframe.loc[start_idx:end_idx, "job_id"] = job_id
            # Append the job to the jobs list
            jobs.append(job)

    return dataframes, jobs


def extract_results_v4(result, job_id: str, qubit_idx: int) -> pd.DataFrame:

    raw_rows = []

    for circuit_idx, res in enumerate(result):
        bitarray = res.data.__dict__[f"c{qubit_idx}"]
        # Get the counts
        counts = bitarray.get_counts()
        num_shots = bitarray.num_shots
        assert isinstance(counts, dict) and isinstance(num_shots, int)

        c_r00 = counts.get("00", 0)  # |unknown>
        c_r01 = counts.get("01", 0)  # |1>
        c_r10 = counts.get("10", 0)  # |0>
        c_r11 = counts.get("11", 0)  # |leakage>

        # Calculate the expectation values
        expval = (c_r10 - c_r01) / (c_r10 + c_r01)
        non_mcmd_expval = (c_r10 + c_r00) - (c_r01 + c_r11)

        raw_rows.append(
            {
                "job_id": job_id,
                "circuit_idx": circuit_idx,
                "unique_id": res.metadata.get("circuit_metadata", {}).get(
                    "unique_id", -1
                ),
                "expectation_value": expval,
                "counts/00": c_r00,
                "counts/01": c_r01,
                "counts/10": c_r10,
                "counts/11": c_r11,
                "status": "COMPLETED",
            }
        )

    return pd.DataFrame(raw_rows)


def extract_results_v3(result, job_id: str, qubit_idx: int) -> pd.DataFrame:

    raw_rows = []

    for circuit_idx, res in enumerate(result):
        bitarray = res.data.__dict__[f"c{qubit_idx}"]
        # Get the counts
        counts = bitarray.get_counts()
        num_shots = (
            bitarray.num_shots
        )  # NOTE: .c because of the name of classical register https://docs.quantum.ibm.com/api/migration-guides/v2-primitives#steps-to-migrate-to-sampler-v2
        assert isinstance(counts, dict) and isinstance(num_shots, int)

        c_r00 = counts.get("00", 0)  # |unknown>
        c_r01 = counts.get("01", 0)  # |1>
        c_r10 = counts.get("10", 0)  # |0>
        c_r11 = counts.get("11", 0)  # |leakage>

        # Calculate the expectation values
        expval = (c_r10 - c_r01) / (c_r10 + c_r01)

        unique_id = res.metadata.get("circuit_metadata", {}).get("unique_id", -1)
        if unique_id == -1:
            unique_id = (
                res.metadata.get("circuit_metadata", {})
                .get(str(qubit_idx), {})
                .get("unique_id", -1)
            )

        raw_rows.append(
            {
                "job_id": job_id,
                "circuit_idx": circuit_idx,
                "unique_id": unique_id,
                "expectation_value": expval,
                "counts/00": c_r00,
                "counts/01": c_r01,
                "counts/10": c_r10,
                "counts/11": c_r11,
                "status": "COMPLETED",
            }
        )

    return pd.DataFrame(raw_rows)


def make_sampler_options(shots: int):
    return SamplerOptions(
        default_shots=shots,
        dynamical_decoupling=DynamicalDecouplingOptions(enable=False),
        twirling=TwirlingOptions(enable_gates=False, enable_measure=False),
    )
