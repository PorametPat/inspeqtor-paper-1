import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
import jax
import jax.numpy as jnp
import typing
import json
import numpy as np
from pathlib import Path
import torch
from torch import Generator
from torch.utils.data import DataLoader, Subset, Dataset
from .typing import ParametersDictType


def add_hilbert_level(op: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Add a level to the operator or state

    Args:
        op (jnp.ndarray): The qubit operator or state
        is_state (bool): True if the operator is a state, False if the operator is an operator

    Returns:
        jnp.ndarray: The qutrit operator or state
    """
    return jax.scipy.linalg.block_diag(op, x)


@dataclass
class Operator:
    _pauli_x = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
    _pauli_y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64)
    _pauli_z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
    _hadamard = jnp.array([[1, 1], [1, -1]], dtype=jnp.complex64) / jnp.sqrt(2)
    _s_gate = jnp.array([[1, 0], [0, 1j]], dtype=jnp.complex64)
    _sdg_gate = jnp.array([[1, 0], [0, -1j]], dtype=jnp.complex64)
    _identity = jnp.array([[1, 0], [0, 1]], dtype=jnp.complex64)

    @classmethod
    def from_label(cls, op: str) -> jnp.ndarray:
        """Initialize the operator from the label

        Args:
            op (str): The label of the operator

        Raises:
            ValueError: Operator not supported

        Returns:
            jnp.ndarray: The operator
        """

        if op == "X":
            operator = cls._pauli_x
        elif op == "Y":
            operator = cls._pauli_y
        elif op == "Z":
            operator = cls._pauli_z
        elif op == "H":
            operator = cls._hadamard
        elif op == "S":
            operator = cls._s_gate
        elif op == "Sdg":
            operator = cls._sdg_gate
        elif op == "I":
            operator = cls._identity
        else:
            raise ValueError(f"Operator {op} is not supported")

        return operator

    @classmethod
    def to_qutrit(cls, op: jnp.ndarray) -> jnp.ndarray:
        return add_hilbert_level(op, x=jnp.array([1.0]))


@dataclass
class State:
    _zero = jnp.array([1, 0], dtype=jnp.complex64)
    _one = jnp.array([0, 1], dtype=jnp.complex64)
    _plus = jnp.array([1, 1], dtype=jnp.complex64) / jnp.sqrt(2)
    _minus = jnp.array([1, -1], dtype=jnp.complex64) / jnp.sqrt(2)
    _right = jnp.array([1, 1j], dtype=jnp.complex64) / jnp.sqrt(2)
    _left = jnp.array([1, -1j], dtype=jnp.complex64) / jnp.sqrt(2)

    @classmethod
    def from_label(cls, state: str, dm: bool = False) -> jnp.ndarray:
        """Initialize the state from the label

        Args:
            state (str): The label of the state
            dm (bool, optional): Initialized as statevector or density matrix. Defaults to False.

        Raises:
            ValueError: State not supported

        Returns:
            jnp.ndarray: The state
        """

        if state in ["0", "Z+"]:
            state_vec = cls._zero
        elif state in ["1", "Z-"]:
            state_vec = cls._one
        elif state in ["+", "X+"]:
            state_vec = cls._plus
        elif state in ["-", "X-"]:
            state_vec = cls._minus
        elif state in ["r", "Y+"]:
            state_vec = cls._right
        elif state in ["l", "Y-"]:
            state_vec = cls._left
        else:
            raise ValueError(f"State {state} is not supported")
        
        state_vec = state_vec.reshape(2, 1)

        return state_vec if not dm else jnp.outer(state_vec, state_vec.conj())

    @classmethod
    def to_qutrit(cls, state: jnp.ndarray) -> jnp.ndarray:
        return add_hilbert_level(state, x=jnp.array([0.0]))


@dataclass
class QubitInformation:
    """Dataclass to store qubit information

    Raises:
        ValueError: Fail to convert unit to GHz
    """

    unit: str
    qubit_idx: int
    anharmonicity: float
    frequency: float
    drive_strength: float
    date: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def __post_init__(self):
        self.convert_unit_to_ghz()

    def convert_unit_to_ghz(self):
        if self.unit == "GHz":
            pass
        elif self.unit == "Hz":
            self.anharmonicity = self.anharmonicity * 1e-9
            self.frequency = self.frequency * 1e-9
            self.drive_strength = self.drive_strength * 1e-9
        elif self.unit == "2piGHz":
            self.anharmonicity = self.anharmonicity / (2 * jnp.pi)
            self.frequency = self.frequency / (2 * jnp.pi)
            self.drive_strength = self.drive_strength / (2 * jnp.pi)
        elif self.unit == "2piHz":
            self.anharmonicity = self.anharmonicity / (2 * jnp.pi) * 1e-9
            self.frequency = self.frequency / (2 * jnp.pi) * 1e-9
            self.drive_strength = self.drive_strength / (2 * jnp.pi) * 1e-9
        else:
            raise ValueError("Unit must be GHz, 2piGHz, 2piHz, or Hz")

        # Set unit to GHz
        self.unit = "GHz"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, dict_qubit_info):
        return cls(**dict_qubit_info)


@dataclass
class ExpectationValue:
    """Dataclass to store expectation value information

    Raises:
        ValueError: Not support initial state
        ValueError: Not support observable
        ValueError: Not support initial state
        ValueError: Not support observable

    Returns:
        _type_: _description_
    """

    initial_state: str
    observable: str
    expectation_value: None | float = None

    # Not serialized
    initial_statevector: jnp.ndarray = field(init=False)
    initial_density_matrix: jnp.ndarray = field(init=False)
    observable_matrix: jnp.ndarray = field(init=False)

    def __post_init__(self):
        if self.initial_state not in ["+", "-", "r", "l", "0", "1"]:
            raise ValueError(f"Initial state {self.initial_state} is not supported")
        if self.observable not in ["X", "Y", "Z"]:
            raise ValueError(f"Observable {self.observable} is not supported")

        self.initial_statevector = State.from_label(self.initial_state)
        self.initial_density_matrix = State.from_label(self.initial_state, dm=True)
        self.observable_matrix = Operator.from_label(self.observable)

    def to_dict(self):
        return {
            "initial_state": self.initial_state,
            "observable": self.observable,
            "expectation_value": self.expectation_value,
        }

    def __eq__(self, __value: object) -> bool:

        if not isinstance(__value, ExpectationValue):
            return False

        return (
            self.initial_state == __value.initial_state
            and self.observable == __value.observable
            and self.expectation_value == __value.expectation_value
        )

    def __str__(self):
        return f"{self.initial_state}/{self.observable} = {self.expectation_value}"

    # Overwrite the __repr__ method of the class
    def __repr__(self):
        return f'{self.__class__.__name__}(initial_state="{self.initial_state}", observable="{self.observable}", expectation_value={self.expectation_value})'

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass
class ExperimentConfiguration:
    """Experiment configuration dataclass"""

    qubits: typing.Sequence[QubitInformation]
    expectation_values_order: typing.Sequence[ExpectationValue]
    parameter_names: typing.Sequence[
        typing.Sequence[str]
    ]  # Get from the pulse sequence .get_parameter_names()
    backend_name: str
    shots: int
    EXPERIMENT_IDENTIFIER: str
    EXPERIMENT_TAGS: typing.Sequence[str]
    description: str
    device_cycle_time_ns: float
    sequence_duration_dt: int
    instance: str
    sample_size: int
    date: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    additional_info: dict[str, str] = field(default_factory=dict)

    def to_dict(self):
        return {
            **asdict(self),
            "qubits": [qubit.to_dict() for qubit in self.qubits],
            "expectation_values_order": [
                exp.to_dict() for exp in self.expectation_values_order
            ],
        }

    @classmethod
    def from_dict(cls, dict_experiment_config):
        dict_experiment_config["qubits"] = [
            QubitInformation.from_dict(qubit)
            for qubit in dict_experiment_config["qubits"]
        ]

        dict_experiment_config["expectation_values_order"] = [
            ExpectationValue.from_dict(exp)
            for exp in dict_experiment_config["expectation_values_order"]
        ]

        return cls(**dict_experiment_config)

    def to_file(self, path: typing.Union[os.PathLike, str]):
        os.makedirs(path, exist_ok=True)
        _path = Path(path)
        with open(_path / "config.json", "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_file(cls, path: typing.Union[os.PathLike, str]):
        _path = Path(path)
        with open(_path / "config.json", "r") as f:
            dict_experiment_config = json.load(f)

        return cls.from_dict(dict_experiment_config)


import pandas as pd  # type: ignore
import logging


@dataclass
class PredefinedCol:
    name: str
    description: str
    type: type
    unique: bool = False
    required: bool = True
    checks: list[typing.Callable[[typing.Union[str, float, int]], bool]] = field(
        default_factory=list
    )


EXPECTATION_VALUE = PredefinedCol(
    name="expectation_value",
    description="The value of the expectation value, must be set later after the experiment.",
    type=float,
    checks=[lambda x: x >= -1 and x <= 1 if isinstance(x, (int, float)) else False],
)

INITIAL_STATE = PredefinedCol(
    name="initial_state",
    description="The initial state (+, -, r, l, 0, 1) of the circuit.",
    type=str,
    checks=[lambda x: x in ["+", "-", "r", "l", "0", "1"]],
)

OBSERVABLE = PredefinedCol(
    name="observable",
    description="The Pauli operator (X, Y, Z) that the circuit measure expectation value on.",
    type=str,
    checks=[lambda x: x in ["X", "Y", "Z"]],
)

PARAMETERS_ID = PredefinedCol(
    name="parameters_id",
    description="The id of the parameters, must be unique for each row of the postprocessed_data.",
    type=int,
)

REQUIRED_COLUMNS = [
    EXPECTATION_VALUE,
    INITIAL_STATE,
    OBSERVABLE,
    PARAMETERS_ID,
]


def make_row(
    expectation_value: float,
    initial_state: str,
    observable: str,
    parameters_id: int,
    parameters_list: list[ParametersDictType],
    **kwargs,
) -> dict[str, typing.Union[float, int, str]]:
    # Prefix the parameters_list keys with f"parameter/{i}/" for i in range(len(parameters_list))
    # and flatten the dict into a single dict
    parameters_dict = {}
    for i, params in enumerate(parameters_list):
        for k, v in params.items():
            parameters_dict[f"parameter/{i}/{k}"] = float(v)

    # Validate that kwargs does not have the same key as parameters_dict and serializable
    assert all([k not in parameters_dict.keys() for k in kwargs.keys()])
    assert all([isinstance(k, str) for k in kwargs.keys()])
    assert all([isinstance(v, (int, float, str)) for v in kwargs.values()])

    return {
        # Require for postprocessing
        EXPECTATION_VALUE.name: expectation_value,
        INITIAL_STATE.name: initial_state,
        OBSERVABLE.name: observable,
        PARAMETERS_ID.name: parameters_id,
        **parameters_dict,
        **kwargs,
    }


def flatten_parameter_name_with_prefix(
    parameter_names: typing.Sequence[typing.Sequence[str]],
) -> list[str]:
    """Create a flatten list of parameter names with prefix parameter/{i}/

    Args:
        parameter_names (typing.Sequence[typing.Sequence[str]]): The list of parameter names from the pulse sequence
                                                                 or the experiment configuration

    Returns:
        list[str]: The flatten list of parameter names with prefix parameter/{i}/
    """
    return [
        f"parameter/{i}/{name}"
        for i, names in enumerate(parameter_names)
        for name in names
    ]


def transform_parameter_name(name: str) -> str:
    if name.startswith("parameter/"):
        return "/".join(name.split("/")[2:])
    else:
        return name


def get_parameters_dict_list(
    parameters_name: typing.Sequence[typing.Sequence[str]], parameters_row: pd.Series
) -> list[ParametersDictType]:
    recovered_parameters = [
        {
            # Split to remove the parameter/{i}/
            transform_parameter_name(k): v
            for k, v in parameters_row.items()
            # Check if the key is parameter/{i}/ and the value is float
            if isinstance(k, str)
            and k.startswith(f"parameter/{i}/")
            and isinstance(v, (float, int))
        }
        for i in range(len(parameters_name))
    ]

    return recovered_parameters


@dataclass
class ExperimentData:
    experiment_config: ExperimentConfiguration
    preprocess_data: pd.DataFrame
    # optional
    _postprocessed_data: pd.DataFrame | None = field(default=None)

    # Setting
    keep_decimal: int = 10
    # Postprocessing
    postprocessed_data: pd.DataFrame = field(init=False)
    parameter_columns: list[str] = field(init=False)
    parameters: np.ndarray = field(init=False)

    def __post_init__(self):
        self.preprocess_data = self.preprocess_data.round(self.keep_decimal)

        # Validate that self.preprocess_data have all the required columns
        self.validate_preprocess_data()
        logging.info("Preprocess data validated")

        if self._postprocessed_data is not None:
            self.postprocessed_data = self._postprocessed_data
            logging.info("Postprocess data set")

        else:
            post_data = self.transform_preprocess_data_to_postprocess_data()
            logging.info("Preprocess data transformed to postprocess data")

            self.postprocessed_data = post_data.round(self.keep_decimal)

        # Validate the data with schema
        self.validate_postprocess_data(self.postprocessed_data)
        logging.info("Postprocess data validated")

        self.parameter_columns = flatten_parameter_name_with_prefix(
            self.experiment_config.parameter_names
        )
        num_features = len(self.experiment_config.parameter_names[0])
        num_pulses = len(self.experiment_config.parameter_names)

        try:
            temp_params = np.array(
                self.postprocessed_data[self.parameter_columns]
                .to_numpy()
                .reshape((self.experiment_config.sample_size, num_pulses, num_features))
            )
        except Exception as e:
            logging.info(
                "Could not reshape parameters with shape (sample_size, num_pulses, num_features), automatically reshaping to (sample_size, -1)"
            )
            temp_params = np.array(
                self.postprocessed_data[self.parameter_columns]
                .to_numpy()
                .reshape((self.experiment_config.sample_size, -1))
            )
            logging.info(f"Parameters reshaped to {temp_params.shape}")

        self.parameters = temp_params

        logging.info("Parameters converted to numpy array")

        assert (
            self.preprocess_data[self.parameter_columns]
            .drop_duplicates(ignore_index=True)
            .equals(
                self.postprocessed_data[self.parameter_columns].drop_duplicates(
                    ignore_index=True
                )
            )
        ), "The preprocess_data and postprocessed_data does not have the same parameters."
        logging.info("Preprocess data and postprocess data have the same parameters")

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, ExperimentData):
            return False

        return (
            self.experiment_config == __value.experiment_config
            and self.preprocess_data.equals(__value.preprocess_data)
        )

    def validate_preprocess_data(self):
        """Validate that the preprocess_data have all the required columns."""
        """
        Required columns:
            - EXPECTATION_VALUE
            - INITIAL_STATE
            - OBSERVABLE
            - PARAMETERS_ID
        """
        for col in REQUIRED_COLUMNS:
            if col.required:
                assert (
                    col.name in self.preprocess_data.columns
                ), f"Column {col.name} is required but not found in the preprocess_data."

        # Validate that the preprocess_data have all expected parameters columns
        required_parameters_columns = flatten_parameter_name_with_prefix(
            self.experiment_config.parameter_names
        )

        for _col in required_parameters_columns:
            assert (
                _col in self.preprocess_data.columns
            ), f"Column {_col} is required but not found in the preprocess_data."

    def validate_postprocess_data(self, post_data: pd.DataFrame):
        logging.info("Validating postprocess data")
        # Validate that the postprocess_data have all the required columns
        for col in REQUIRED_COLUMNS:
            if col.required:
                assert (
                    col.name in post_data.columns
                ), f"Column {col.name} is required but not found in the postprocess_data."

        # Validate the check functions
        for col in REQUIRED_COLUMNS:
            for check in col.checks:
                assert all(
                    [check(v) for v in post_data[col.name]]
                ), f"Column {col.name} failed the check function {check}"

        # Validate that the postprocess_data have all expected parameters columns
        required_parameters_columns = flatten_parameter_name_with_prefix(
            self.experiment_config.parameter_names
        )
        for _col in required_parameters_columns:
            assert (
                _col in post_data.columns
            ), f"Column {_col} is required but not found in the postprocess_data."

    def transform_preprocess_data_to_postprocess_data(self) -> pd.DataFrame:
        # Postprocess the data squeezing the data into the expectation values
        # Required columns: PARAMETERS_ID, OBSERVABLE, INITIAL_STATE, EXPECTATION_VALUE, + experiment_config.parameter_names
        post_data = []

        for params_id in range(self.experiment_config.sample_size):
            # NOTE: Assume that parameters_id starts from 0 and is continuous to sample_size - 1
            rows = self.preprocess_data.loc[
                self.preprocess_data[PARAMETERS_ID.name] == params_id
            ]

            expectation_values = {}
            for _, exp_order in enumerate(
                self.experiment_config.expectation_values_order
            ):
                expectation_value = rows.loc[
                    (rows[OBSERVABLE.name] == exp_order.observable)
                    & (rows[INITIAL_STATE.name] == exp_order.initial_state)
                ][EXPECTATION_VALUE.name].values

                if expectation_value.shape[0] != 1:
                    raise ValueError(
                        f"Expectation value for params_id {params_id}, initial_state {exp_order.initial_state}, observable {exp_order.observable} is not unique. The length is {len(expectation_value)}."
                    )

                expectation_values[
                    f"{EXPECTATION_VALUE.name}/{exp_order.initial_state}/{exp_order.observable}"
                ] = expectation_value[0]

            drop_duplicates_row = rows.drop_duplicates(
                subset=flatten_parameter_name_with_prefix(self.experiment_config.parameter_names)
            )
            # Assert that only one row is returned
            assert drop_duplicates_row.shape[0] == 1
            pulse_parameters = drop_duplicates_row.to_dict(orient="records")[0]

            new_row = {
                PARAMETERS_ID.name: params_id,
                **expectation_values,
                **{str(k): v for k, v in pulse_parameters.items()},
            }

            post_data.append(new_row)

        return pd.DataFrame(post_data)

    def get_parameters_dataframe(self) -> pd.DataFrame:
        return self.postprocessed_data[self.parameter_columns]

    def get_expectation_values(self) -> np.ndarray:

        expectation_value = self.postprocessed_data[
            [
                f"expectation_value/{col.initial_state}/{col.observable}"
                for col in self.experiment_config.expectation_values_order
            ]
        ].to_numpy()

        return np.array(expectation_value)

    def get_parameters_dict_list(self) -> list[list[ParametersDictType]]:
        _temp = self.postprocessed_data[self.parameter_columns]

        _params_list = [
            get_parameters_dict_list(self.experiment_config.parameter_names, row)
            for _, row in _temp.iterrows()
        ]

        return _params_list

    def save_to_folder(self, path: typing.Union[os.PathLike, str]):
        os.makedirs(path, exist_ok=True)
        self.experiment_config.to_file(path)
        self.preprocess_data.to_csv(f"{path}/preprocess_data.csv", index=False)
        self.postprocessed_data.to_csv(f"{path}/postprocessed_data.csv", index=False)

    @classmethod
    def from_folder(cls, path: typing.Union[os.PathLike, str]) -> "ExperimentData":
        _path = Path(path)
        experiment_config = ExperimentConfiguration.from_file(_path)
        preprocess_data = pd.read_csv(
            # f"{path}/preprocess_data.csv",
            _path / "preprocess_data.csv",
        )

        # Check if postprocessed_data exists
        if not os.path.exists(f"{str(path)}/postprocessed_data.csv"):
            postprocessed_data = None
        else:
            postprocessed_data = pd.read_csv(
                # f"{path}/postprocessed_data.csv",
                _path / "postprocessed_data.csv",
            )

        return cls(
            experiment_config=experiment_config,
            preprocess_data=preprocess_data,
            _postprocessed_data=postprocessed_data,
        )

    def analysis_sum_of_expectation_values(self) -> pd.DataFrame:

        paulis = ["X", "Y", "Z"]
        initial_states = [("0", "1"), ("+", "-"), ("r", "l")]
        data = {}
        for pauli in paulis:
            for initial_state in initial_states:
                _name = f"{pauli}/{initial_state[0]}/{initial_state[1]}"

                res = (
                    self.postprocessed_data[
                        f"expectation_value/{initial_state[0]}/{pauli}"
                    ]
                    + self.postprocessed_data[
                        f"expectation_value/{initial_state[1]}/{pauli}"
                    ]
                )

                data[_name] = res.to_numpy()

        return pd.DataFrame(data)


def generate_path_with_datetime(sub_dir: typing.Union[str, None] = None):
    return os.path.join(
        sub_dir if sub_dir is not None else "",
        datetime.now().strftime("Y%YM%mD%d-H%HM%MS%S"),
    )


@dataclass
class DataConfig:
    EXPERIMENT_IDENTIFIER: str
    hamiltonian: str
    pulse_sequence: dict

    def to_file(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/data_config.json", "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    @classmethod
    def from_file(cls, path: str):
        with open(f"{path}/data_config.json", "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class SpecQDataset(Dataset):
    def __init__(
        self,
        pulse_parameters: np.ndarray,
        unitaries: np.ndarray,
        expectation_values: np.ndarray,
    ):
        self.pulse_parameters = torch.from_numpy(pulse_parameters)
        self.unitaries = torch.from_numpy(unitaries)
        self.expectation_values = torch.from_numpy(expectation_values)

    def __len__(self):
        return self.pulse_parameters.shape[0]

    def __getitem__(self, idx):
        return {
            # Pulse parameters
            "x0": self.pulse_parameters[idx],
            # Unitaries
            "x1": self.unitaries[idx],
            # Expectation values
            "y": self.expectation_values[idx],
        }


def prepare_dataset(
    pulse_parameters: jnp.ndarray,
    unitaries: jnp.ndarray,
    expectation_values: jnp.ndarray,
    key: jnp.ndarray,
    batch_size_ratio: float = 0.1,
    train_val_ratio: float = 0.2,
):

    # Sanity check on the inputs
    # pulse_parameters.shape should be len(shape) == 2
    assert (
        len(pulse_parameters.shape) == 2
    ), "The shape of pulse parameters should be (batch, pulse_params)"
    pulse_parameters_size = pulse_parameters.shape[0]

    # unitaries.shape should be (batch, 2, 2)
    if len(unitaries.shape) == 4:
        _unitaries = unitaries[:, -1, :, :]
        logging.info(
            "Assumed that unitaries provided has time step dims at axis = 1, select only the last time step"
        )
    elif len(unitaries.shape) == 3:
        _unitaries = jnp.array(unitaries)
    else:
        raise ValueError(
            f"The provided unitaries is invalid with shape {str(unitaries.shape)}"
        )
    unitaries_size = unitaries.shape[0]

    assert (
        len(expectation_values.shape) == 2
    ), "The shape of pulse parameters should be (batch, expectation_value_order)"
    expectation_values_size = expectation_values.shape[0]
    # Check if we needed to transpose the unitaries or not.
    if not expectation_values_size == pulse_parameters_size:
        # Try to transpose
        _expectation_value = np.array(expectation_values.T)
        expectation_values_size = _expectation_value.shape[0]
    else:
        _expectation_value = np.array(expectation_values)

    # Check if all of the sizes are the same
    assert (
        expectation_values_size == unitaries_size
    ), "Batch size of the pulse_parameters, expectation_values, unitaries are not the same"

    g = Generator()
    g.manual_seed(0)

    dataset = SpecQDataset(
        pulse_parameters=np.array(pulse_parameters),
        unitaries=np.array(_unitaries),  # only the final unitary
        expectation_values=np.array(_expectation_value),
    )

    assert (train_val_ratio * len(dataset)) % 1 == 0, "The train_val_ratio is invalid, "

    assert (
        train_val_ratio > batch_size_ratio
    ), "batch_size_ratio is larger than train_val_ratio"

    assert (
        int(train_val_ratio % batch_size_ratio) == 0
    ), "train_val_ratio must be multiplier of batch_size_ratio"

    # Randomly split dataset into training and validation
    val_indices = jax.random.choice(
        key, len(dataset), (int(train_val_ratio * len(dataset)),), replace=False
    ).tolist()

    training_indices = list(set([i for i in range(len(dataset))]) - set(val_indices))

    train_dataset = Subset(dataset, training_indices)
    val_dataset = Subset(dataset, val_indices)

    _batch_size = int(pulse_parameters.shape[0] * batch_size_ratio)

    train_dataloader = DataLoader(
        train_dataset, batch_size=_batch_size, shuffle=True, generator=g
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=_batch_size, shuffle=True, generator=g
    )

    return train_dataloader, val_dataloader


def extract_data(
    dataloader: DataLoader,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    
    # Assert that the dataloader has dataset attribute and the dataset has dataset attribute
    assert hasattr(dataloader, "dataset"), "The dataloader does not have dataset attribute"
    assert hasattr(dataloader.dataset, "dataset"), "The dataloader.dataset does not have dataset attribute"

    return (
        dataloader.dataset.dataset.pulse_parameters,
        dataloader.dataset.dataset.unitaries,
        dataloader.dataset.dataset.expectation_values,
    )
