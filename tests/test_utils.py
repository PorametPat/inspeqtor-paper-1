import jax.numpy as jnp
from specq_jax import core
from specq_jax.simulator import tensor

def test_tensor():

    assert jnp.allclose(tensor(core.X, 0, 2), jnp.kron(core.X, jnp.eye(2)))

    assert jnp.allclose(tensor(core.X, 1, 2), jnp.kron(jnp.eye(2), core.X))

    assert jnp.allclose(tensor(core.X, 0, 3), jnp.kron(core.X, jnp.eye(4)))

    assert jnp.allclose(tensor(core.X, 1, 3), jnp.kron(jnp.eye(2), jnp.kron(core.X, jnp.eye(2))))

    # Mixed operators
    assert jnp.allclose(tensor(core.X, 0, 2) @ tensor(core.Y, 1, 2), jnp.kron(core.X, core.Y))