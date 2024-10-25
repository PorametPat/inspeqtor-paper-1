import jax
import jax.numpy as jnp
import specq_jax as sqj
import specq_dev as sqd
import numpyro
from numpyro.contrib.module import random_flax_module
import numpyro.distributions as dist
from numpyro.infer import Predictive, SVI, TraceMeanField_ELBO, autoguide, init_to_feasible
from functools import partial
import optax
from pathlib import Path
print(1)