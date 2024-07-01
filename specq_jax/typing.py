import jax.numpy as jnp
from typing import Callable, Union, Protocol, Any
from flax.typing import VariableDict
from jaxtyping import Array, Complex, Float
import optax # type: ignore
from dataclasses import dataclass


class LossFn(Protocol):

    def __call__(
        self,
        params: VariableDict,
        pulse_parameters: Float[Array, "batch num_pulses num_features"],  # noqa: F722
        unitaries: Complex[Array, "batch dim dim"],  # noqa: F722
        expectation_values: Float[Array, "batch num_expectations"],  # noqa: F722
        training: bool,
    ) -> Float[Array, "1"]: ...


class TrainStepFn(Protocol):

    def __call__(
        self,
        params: VariableDict,
        opt_state: optax.OptState,
        pulse_parameters: Float[Array, "batch num_pulses num_features"],  # noqa: F722
        unitaries: Complex[Array, "batch dim dim"],  # noqa: F722
        expectations: Float[Array, "batch num_expectations"],  # noqa: F722
        dropout_key: jnp.ndarray,
        transform_state: Union[None, optax.OptState],
    ) -> tuple[Any, Any, float]: ...


class TestStepFn(Protocol):

    def __call__(
        self,
        params: VariableDict,
        pulse_parameters: Float[Array, "batch num_pulses num_features"],  # noqa: F722
        unitaries: Complex[Array, "batch dim dim"],  # noqa: F722
        expectations: Float[Array, "batch num_expectations"],  # noqa: F722
    ) -> float: ...


@dataclass
class HistoryEntry:
    epoch: int
    step: int
    batch_loss: float
    global_step: int
    epoch_loss: float | None = None
    val_loss: float | None = None
    lr: float | None = None
    


class CallbackFn(Protocol):
    def __call__(
        self,
        params: VariableDict,
        opt_state: optax.OptState,
        history: list[HistoryEntry],
    ) -> None: ...