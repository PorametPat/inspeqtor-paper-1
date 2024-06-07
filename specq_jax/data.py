import specq_dev.shared as specq  # type: ignore
from specq_dev.jax.pulse import construct_pulse_sequence_reader, JaxBasedPulseSequence  # type: ignore
import jax.numpy as jnp

import jax
from typing import Callable
import logging
import numpy as np

from dataclasses import dataclass, asdict
import os
import datetime
import json
from typing import Union, Callable
import pandas as pd
from .pulse import MultiDragPulse, DragPulse, MultiDragPulseV2
from specq_dev.test_utils import JaxDragPulse  # type: ignore
import torch
from torch.utils.data import Dataset

from os import PathLike
from .simulator import simulator as sqjs_simulator, signal_func_v3, SignalParameters
from functools import partial
from .utils.helper import rotating_transmon_hamiltonian

from torch import Generator
from torch.utils.data import DataLoader, Subset

def generate_path_with_datetime(sub_dir: Union[str, None] = None):
    return os.path.join(
        sub_dir if sub_dir is not None else "",
        datetime.datetime.now().strftime("Y%YM%mD%d-H%HM%MS%S"),
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


@dataclass
class ModelState:
    model_config: dict
    model_params: dict

    def save(self, path: str):
        # turn the dataclass into a dictionary
        model_params = jax.tree.map(lambda x: x.tolist(), self.model_params)

        os.makedirs(path, exist_ok=True)

        with open(f"{path}/model_config.json", "w") as f:
            json.dump(self.model_config, f, indent=4)

        with open(f"{path}/model_params.json", "w") as f:
            json.dump(model_params, f, indent=4)

    @classmethod
    def load(cls, path: str):
        with open(f"{path}/model_config.json", "r") as f:
            model_config = json.load(f)

        with open(f"{path}/model_params.json", "r") as f:
            model_params = json.load(f)

        # Define is_leaf function to check if a node is a leaf
        # If the node is a leaf, convert it to a jnp.array
        # The function will check if object is list, if list then convert to jnp.array
        def is_leaf(x):
            return isinstance(x, list)

        # Apply the inverse of the tree.map function
        model_params = jax.tree.map(
            lambda x: jnp.array(x), model_params, is_leaf=is_leaf
        )

        return cls(model_config, model_params)


def save_model(
    path: str,
    experiment_identifier: str,
    pulse_sequence: specq.BasePulseSequence,
    hamiltonian: Union[str, Callable],
    model_config: dict,
    model_params: dict,
    history: list,
    with_auto_datetime: bool = True,
):

    _path = generate_path_with_datetime(path) if with_auto_datetime else path

    os.makedirs(_path, exist_ok=True)

    model_state = ModelState(
        model_config=model_config,
        model_params=model_params,
    )

    model_state.save(_path + "/model_state")

    # Save the history
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(_path + "/history.csv", index=False)

    # Save the data config
    data_config = DataConfig(
        EXPERIMENT_IDENTIFIER=experiment_identifier,
        hamiltonian=(
            hamiltonian if isinstance(hamiltonian, str) else hamiltonian.__name__
        ),
        pulse_sequence=pulse_sequence.to_dict(),
    )

    data_config.to_file(_path)

    return _path


def load_model(path: str):
    model_state = ModelState.load(path + "/model_state")
    hist_df = pd.read_csv(path + "/history.csv")
    data_config = DataConfig.from_file(path)

    return (
        model_state,
        hist_df.to_dict(orient="records"),
        data_config,
    )


pulse_reader = construct_pulse_sequence_reader(
    pulses=[MultiDragPulse, DragPulse, JaxDragPulse, MultiDragPulseV2]
)


def load_data(path: PathLike):

    # Load the data from the experiment
    exp_data = specq.ExperimentData.from_folder(path)
    dt = exp_data.experiment_config.device_cycle_time_ns
    qubit_info = exp_data.experiment_config.qubits[0]

    logging.info(f"Loaded data from {exp_data.experiment_config.EXPERIMENT_IDENTIFIER}")

    pulse_sequence = pulse_reader(path)

    t_eval = jnp.linspace(
        0, pulse_sequence.pulse_length_dt * dt, pulse_sequence.pulse_length_dt
    )

    hamiltonian = partial(
        rotating_transmon_hamiltonian,
        qubit_info=qubit_info,
        signal=signal_func_v3(
            pulse_sequence.get_envelope,
            qubit_info.frequency,
            dt,
        ),
    )

    jiited_simulator = jax.jit(
        partial(
            sqjs_simulator,
            t_eval=t_eval,
            hamiltonian=hamiltonian,
            y0=jnp.eye(2, dtype=jnp.complex64),
        )
    )

    signal_params: list[SignalParameters] = []
    for pulse_params in exp_data.get_parameters_dict_list():
        signal_params.append(SignalParameters(pulse_params=pulse_params, phase=0))

    unitaries = []
    for signal_param in signal_params:
        unitaries.append(jiited_simulator(signal_param))

    np_unitaries = np.array(unitaries)[:, -1, :, :]

    logging.info(
        f"Finished preparing the data for the experiment {exp_data.experiment_config.EXPERIMENT_IDENTIFIER}"
    )

    pulse_parameters = exp_data.parameters
    expectations = exp_data.get_expectation_values()

    return (
        exp_data,
        pulse_parameters,
        np_unitaries,
        expectations,
        pulse_sequence,
        jiited_simulator,
    )


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

    assert train_val_ratio > batch_size_ratio, "batch_size_ratio is larger than train_val_ratio"

    assert int(train_val_ratio % batch_size_ratio) == 0, "train_val_ratio must be multiplier of batch_size_ratio"

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