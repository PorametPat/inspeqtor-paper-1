import jax.numpy as jnp
import specq_dev.shared as specq # type: ignore 

X, Y, Z = (
    jnp.array(specq.Operator.from_label("X")),
    jnp.array(specq.Operator.from_label("Y")),
    jnp.array(specq.Operator.from_label("Z")),
)