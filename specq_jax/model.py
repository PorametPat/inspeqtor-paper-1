from typing import Callable, Sequence
from flax import linen as nn
import jax.numpy as jnp


class BasicBlackBox(nn.Module):
    feature_size: int
    hidden_sizes_1: Sequence[int] = (20, 10)
    hidden_sizes_2: Sequence[int] = (20, 10)
    pauli_operators: Sequence[str] = ("X", "Y", "Z")

    NUM_UNITARY_PARAMS: int = 3
    NUM_DIAGONAL_PARAMS: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(features=self.feature_size)(x)
        # Flatten the input
        x = x.reshape((x.shape[0], -1))
        # Apply a activation function
        x = nn.relu(x)
        # Dropout
        # x = nn.Dropout(0.2)(x)
        # Apply a dense layer for each hidden size
        for hidden_size in self.hidden_sizes_1:
            x = nn.Dense(features=hidden_size)(x)
            x = nn.relu(x)

        Wos_params: dict[str, dict[str, jnp.ndarray]] = dict()
        for op in self.pauli_operators:
            # Sub hidden layer
            for hidden_size in self.hidden_sizes_2:
                _x = nn.Dense(features=hidden_size)(x)
                _x = nn.relu(_x)

            Wos_params[op] = dict()
            # For the unitary part, we use a dense layer with 3 features
            unitary_params = nn.Dense(features=self.NUM_UNITARY_PARAMS)(_x)
            # Apply sigmoid to this layer
            unitary_params = 2 * jnp.pi * nn.sigmoid(unitary_params)
            # For the diagonal part, we use a dense layer with 1 feature
            diag_params = nn.Dense(features=self.NUM_DIAGONAL_PARAMS)(_x)
            # Apply the activation function
            diag_params = nn.tanh(diag_params)

            Wos_params[op] = {
                "U": unitary_params,
                "D": diag_params,
            }

        return Wos_params


class ParallelBlackBox(nn.Module):
    hidden_sizes: Sequence[int] = (20, 10)
    pauli_operators: Sequence[str] = ("X", "Y", "Z")

    activation: Callable = nn.tanh
    NUM_UNITARY_PARAMS: int = 3
    NUM_DIAGONAL_PARAMS: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray):

        Wos_params: dict[str, dict[str, jnp.ndarray]] = dict()
        for op in self.pauli_operators:
            # Sub hidden layer
            for hidden_size in self.hidden_sizes:
                _x = nn.Dense(features=hidden_size)(x)
                _x = nn.relu(_x)

            Wos_params[op] = dict()
            # For the unitary part, we use a dense layer with 3 features
            unitary_params = nn.Dense(features=self.NUM_UNITARY_PARAMS)(_x)
            # Apply sigmoid to this layer
            unitary_params = 2 * jnp.pi * nn.sigmoid(unitary_params)
            # For the diagonal part, we use a dense layer with 1 feature
            diag_params = nn.Dense(features=self.NUM_DIAGONAL_PARAMS)(_x)
            # Apply the activation function
            diag_params = self.activation(diag_params)

            Wos_params[op] = {
                "U": unitary_params,
                "D": diag_params,
            }

        return Wos_params

class GRUBlackBox(nn.Module):

    @nn.compact
    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)