import os
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from .typing import ParametersDictType
import typing
import json
import jax
import jax.numpy as jnp
from .visualization import draw_complex_pulse
from matplotlib.axes import Axes


@dataclass
class BasePulse(ABC):
    def __post_init__(self):
        self.validate()

    @abstractmethod
    def validate(self) -> None: ...

    @abstractmethod
    def get_waveform(self, params: ParametersDictType): ...

    @abstractmethod
    def sample_params(self, *arg, **kwarg) -> ParametersDictType: ...

    @abstractmethod
    def get_bounds(
        self, *arg, **kwarg
    ) -> tuple[ParametersDictType, ParametersDictType]: ...

    def get_envelope(self, params: ParametersDictType) -> typing.Callable:
        raise NotImplementedError("get_envelopes method is not implemented")

    def to_dict(self) -> dict[str, typing.Union[int, float, str]]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass
class BasePulseSequence(ABC):
    pulses: typing.Sequence[BasePulse]
    pulse_length_dt: int

    def __post_init__(self):
        # validate that each pulse have len of pulse_length_dt
        self.validate()

    @abstractmethod
    def validate(self) -> None: ...

    # Must check that the sum of the pulse lengths is equal to the total length of the pulse sequence

    @abstractmethod
    def sample_params(self, *arg, **kwarg) -> list[ParametersDictType]: ...

    @abstractmethod
    def get_waveform(self, params_list: list[ParametersDictType]):
        """
        Samples the pulse sequence by generating random parameters for each pulse and computing the total waveform.

        Parameters:
            key (Key): The random key used for generating the parameters.

        Returns:
            tuple[list[ParametersDictType], np.ndarray]: A tuple containing a list of parameter dictionaries for each pulse and the total waveform.

        Example:
            rng = np.random.default_rng(0)
            params_list, total_waveform = sample(key)
            # params_list: List of parameter dictionaries for each pulse
            # total_waveform: The computed total waveform
        """
        ...

    def get_envelope(self, params_list: list[ParametersDictType]) -> typing.Callable:

        callables = []
        for _params, _pulse in zip(params_list, self.pulses):
            callables.append(_pulse.get_envelope(_params))

        # Create a function that returns the sum of the envelopes
        def envelope(t):
            return sum([c(t) for c in callables])

        return envelope

    def get_bounds(self) -> tuple[list[ParametersDictType], list[ParametersDictType]]:

        lower_bounds = []
        upper_bounds = []
        for pulse in self.pulses:
            lower, upper = pulse.get_bounds()
            lower_bounds.append(lower)
            upper_bounds.append(upper)

        return lower_bounds, upper_bounds

    @abstractmethod
    def get_parameter_names(self) -> list[list[str]]: ...

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            **asdict(self),
            "pulses": [
                {**pulse.to_dict(), "_name": pulse.__class__.__name__}
                for pulse in self.pulses
            ],
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, typing.Any], pulses: typing.Sequence[type[BasePulse]]
    ) -> "BasePulseSequence":

        parsed_data = []
        for d, pulse in zip(data["pulses"], pulses):
            assert isinstance(d, dict), f"Expected dict, got {type(d)}"

            # remove the _name key
            d.pop("_name")
            parsed_data.append(pulse.from_dict(d))

        data["pulses"] = parsed_data

        return cls(**data)

    def to_file(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/pulse_sequence.json", "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_file(
        cls, path: str, pulses: typing.Sequence[type[BasePulse]]
    ) -> "BasePulseSequence":
        with open(f"{path}/pulse_sequence.json", "r") as f:
            dict_pulse_sequence = json.load(f)

        return cls.from_dict(dict_pulse_sequence, pulses=pulses)

    def draw(
        self,
        params_list: list[ParametersDictType],
        ax: typing.Union[None, Axes] = None,
        font_size: int = 12,
    ):

        x_axis = jnp.arange(self.pulse_length_dt)
        waveform = jnp.array(self.get_waveform(params_list))

        fig, ax = draw_complex_pulse(waveform, x_axis, ax=ax, font_size=font_size)

        return fig, ax


@dataclass
class JaxBasedPulse(BasePulse):
    def __post_init__(self):
        self.validate()

    def validate(self):
        # Validate that all attributes are json serializable
        try:
            json.dumps(self.to_dict())
        except TypeError as e:
            raise TypeError(
                f"Cannot serialize {self.__class__.__name__} to json"
            ) from e

        # Validate that the sampling function is working
        key = jax.random.PRNGKey(0)
        params = self.sample_params(key)
        waveform = self.get_waveform(params)

        assert all(
            [isinstance(k, str) for k in params.keys()]
        ), "All key of params dict must be string"
        assert all(
            [isinstance(v, float) for v in params.values()]
        ), "All value of params dict must be jax.Array"
        assert isinstance(waveform, jax.Array), "Waveform must be jax.Array"

        # Validate that params is serializable and deserializable
        try:
            reread_params = json.loads(json.dumps(params))
            assert params == reread_params

        except TypeError as e:
            raise TypeError(
                f"Cannot serialize params dict of {self.__class__.__name__} to json"
            ) from e


@dataclass
class JaxBasedPulseSequence(BasePulseSequence):
    pulses: typing.Sequence[JaxBasedPulse]
    pulse_length_dt: int

    def validate(self):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, self.pulse_length_dt)
        for _key, _pulse in zip(subkeys, self.pulses):
            params = _pulse.sample_params(_key)
            waveform = _pulse.get_waveform(params)
            assert isinstance(waveform, jax.Array)
            # Assert the waveform is of the correct length
            assert waveform.shape == (self.pulse_length_dt,)
            # Assert that all key of params dict is string and all value is jax.Array
            assert all(
                [isinstance(k, str) for k in params.keys()]
            ), "All key of params dict must be string"
            assert all(
                [isinstance(v, (float, int)) for v in params.values()]
            ), "All value of params dict must be jax.Array"

        params = self.sample_params(key)

        # Assert that the bounds have the same pytree structure as the parameters
        lower, upper = self.get_bounds()

        assert jax.tree.structure(lower) == jax.tree.structure(params)
        assert jax.tree.structure(upper) == jax.tree.structure(params)

    def sample_params(self, key: jax.Array) -> list[ParametersDictType]:

        # Split key for each pulse
        subkeys = jax.random.split(key, self.pulse_length_dt)

        params_list: list[ParametersDictType] = []
        for _key, _pulse in zip(subkeys, self.pulses):
            params = _pulse.sample_params(_key)
            params_list.append(params)

        return params_list

    def get_waveform(self, params_list: list[ParametersDictType]) -> jnp.ndarray:
        """
        Samples the pulse sequence by generating random parameters for each pulse and computing the total waveform.

        Parameters:
            key (Key): The random key used for generating the parameters.

        Returns:
            tuple[list[ParametersDictType], Complex[Array, "time"]]: A tuple containing a list of parameter dictionaries for each pulse and the total waveform.

        Example:
            key = jax.random.PRNGKey(0)
            params_list, total_waveform = sample(key)
            # params_list: List of parameter dictionaries for each pulse
            # total_waveform: The computed total waveform
        """
        # Create base waveform
        total_waveform = jnp.zeros(self.pulse_length_dt, dtype=jnp.complex64)

        for _params, _pulse in zip(params_list, self.pulses):
            waveform = _pulse.get_waveform(_params)
            total_waveform += waveform

        return total_waveform

    def get_parameter_names(self) -> list[list[str]]:
        # Sample the pulse sequence to get the parameter names
        key = jax.random.PRNGKey(0)
        params_list = self.sample_params(key)

        # Get the parameter names for each pulse
        parameter_names = []
        for params in params_list:
            parameter_names.append(list(params.keys()))

        return parameter_names


def array_to_list_of_params(array: jnp.ndarray, parameter_structure: list[list[str]]):

    temp = []
    idx = 0
    for sub_pulse in parameter_structure:
        temp_dict = {}
        for param in sub_pulse:
            temp_dict[param] = array[idx]
            idx += 1
        temp.append(temp_dict)

    return temp


def list_of_params_to_array(
    params: list[ParametersDictType], parameter_structure: list[list[str]]
):

    temp = []
    for subp_idx, sub_pulse in enumerate(parameter_structure):
        for param in sub_pulse:
            temp.append(params[subp_idx][param])

    return jnp.array(temp)


def construct_pulse_sequence_reader(
    pulses: list[type[BasePulse]] = [],
) -> typing.Callable[[str], BasePulseSequence]:

    default_pulses: list[type[BasePulse]] = []

    # Merge the default pulses with the provided pulses
    pulses_list = default_pulses + pulses

    def pulse_sequence_reader(path) -> BasePulseSequence:
        with open(f"{path}/pulse_sequence.json", "r") as f:
            pulse_sequence_dict = json.load(f)

        parsed_pulses = []

        for pulse_dict in pulse_sequence_dict["pulses"]:
            for pulse_class in pulses_list:
                if pulse_dict["_name"] == pulse_class.__name__:
                    parsed_pulses.append(pulse_class)

        return JaxBasedPulseSequence.from_dict(
            pulse_sequence_dict, pulses=parsed_pulses
        )

    return pulse_sequence_reader
