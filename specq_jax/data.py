import specq_dev.shared as specq  # type: ignore
import specq_dev.jax as specq_jax  # type: ignore
from specq_dev.jax.pulse import construct_pulse_sequence_reader # type: ignore
from specq_jax.core import get_simulator, rotating_duffling_oscillator_hamiltonian
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
from .pulse import MultiDragPulse, DragPulse
from specq_dev.test_utils import JaxDragPulse # type: ignore

def generate_path_with_datetime(sub_dir: Union[str, None] = None):
    return os.path.join(
        sub_dir if sub_dir is not None else "",
        datetime.datetime.now().strftime("Y%YM%mD%d-H%HM%MS%S")
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
        model_params = jax.tree.map(lambda x: jnp.array(x), model_params, is_leaf=is_leaf)

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
        hamiltonian=hamiltonian if isinstance(hamiltonian, str) else hamiltonian.__name__,
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

def calculate_unitaries(
    pulse_sequence: specq.BasePulseSequence,
):

    return


pulse_reader = construct_pulse_sequence_reader(
    pulses=[
        MultiDragPulse, DragPulse, JaxDragPulse
    ]
)

def load_data(
    path: str,
    hamiltonian=rotating_duffling_oscillator_hamiltonian,
):
    # Load the data from the experiment
    exp_data = specq.ExperimentData.from_folder(path)

    logging.info(f"Loaded data from {exp_data.experiment_config.EXPERIMENT_IDENTIFIER}")

    # Setup the simulator
    dt = exp_data.experiment_config.device_cycle_time_ns
    qubit_info = exp_data.experiment_config.qubits[0]
    # pulse_sequence = get_pulse_sequence()
    pulse_sequence = pulse_reader(path)
    t_eval = jnp.linspace(
        0, pulse_sequence.pulse_length_dt * dt, pulse_sequence.pulse_length_dt
    )
    simulator = get_simulator(
        qubit_info=qubit_info, t_eval=t_eval, hamiltonian=hamiltonian
    )

    # Get the waveforms for each pulse parameters to get the unitaries
    _waveforms = []
    for pulse_params in exp_data.get_parameters_dict_list():
        _waveforms.append(pulse_sequence.get_waveform(pulse_params))

    waveforms = jnp.array(_waveforms)

    logging.info(
        f"Prepared the waveforms for the experiment {exp_data.experiment_config.EXPERIMENT_IDENTIFIER}"
    )

    # Check if the unitaries are already calculated
    # if os.path.exists(f"{path}/unitaries.npy"):
    #     unitaries = np.load(f"{path}/unitaries.npy")
    #     logging.info(
    #         f"Loaded the unitaries for the experiment {exp_data.experiment_config.EXPERIMENT_IDENTIFIER}"
    #     )
    # else:

    # jit the simulator
    jitted_simulator = jax.jit(simulator)
    # batch the simulator
    batched_simulator = jax.vmap(jitted_simulator, in_axes=(0))
    # Get the unitaries
    unitaries = batched_simulator(waveforms)

    logging.info(
        f"Got the unitaries for the experiment {exp_data.experiment_config.EXPERIMENT_IDENTIFIER}"
    )

    # Get the final unitaries
    unitaries = np.array(unitaries[:, -1, :, :])
        # Save the unitaries
        # np.save(f"{path}/unitaries.npy", unitaries)

    # Get the expectation values from the experiment
    expectations = exp_data.get_expectation_values()
    # Get the pulse parameters
    pulse_parameters = exp_data.parameters

    logging.info(
        f"Finished preparing the data for the experiment {exp_data.experiment_config.EXPERIMENT_IDENTIFIER}"
    )

    return (
        exp_data,
        pulse_parameters,
        unitaries,
        expectations,
        pulse_sequence,
        simulator,
    )
