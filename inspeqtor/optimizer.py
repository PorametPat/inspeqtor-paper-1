import jax
import jax.numpy as jnp
import typing
from jaxopt import ProjectedGradient  # type: ignore
from jaxopt.projection import projection_box  # type: ignore
import optax  # type: ignore
import flax.linen as nn  # type: ignore
from flax.typing import VariableDict  # type: ignore
import tempfile
from enum import Enum
from ray import tune, train
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import Searcher
import typing
from .typing import ParametersDictType, HistoryEntry, LossChoice
from .pulse import JaxBasedPulseSequence
from .data import prepare_dataset
from .utils.predefined import rotating_transmon_hamiltonian
from .model import (
    default_adaptive_lr_transform,
    BasicBlackBox,
    MLPBlackBox,
    save_model,
    load_model,
    train_model,
)
from .model_v2 import train_model as train_model_v2, HistoryEntryV2, MAEF_loss_fn


def optimize(
    x0: jnp.ndarray,
    lower: list[ParametersDictType],
    upper: list[ParametersDictType],
    fun: typing.Callable[[jnp.ndarray], jnp.ndarray],
    max_iter: int = 1000,
):

    pg = ProjectedGradient(fun=fun, projection=projection_box, maxiter=max_iter)
    opt_params, state = pg.run(jnp.array(x0), hyperparams_proj=(lower, upper))

    return opt_params, state


def optimize_v2(x0, lower, upper, func, optimizer, maxiter=1000):

    params = x0
    opt_state = optimizer.init(params)

    losses = []
    performances = []

    for _ in range(maxiter):
        (loss, aux), grads = jax.value_and_grad(func, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        # Apply projection
        params = optax.projections.projection_box(params, lower, upper)

        losses.append(loss)
        performances.append(aux)

    return params, losses, performances


def default_trainable(
    pulse_sequence: JaxBasedPulseSequence,
    with_dropout: bool = False,
    loss_type: LossChoice = LossChoice.MSEE,
):
    def trainable(
        config: dict,
        pulse_parameters: jnp.ndarray,
        unitaries: jnp.ndarray,
        expectations: jnp.ndarray,
        model_key: jnp.ndarray,
        train_key: jnp.ndarray,
    ):

        FEATURE_SIZE = config["feature_size"]
        HIDDEN_LAYER_1_1 = config["hidden_layer_1_1"]
        HIDDEN_LAYER_1_2 = config["hidden_layer_1_2"]
        HIDDEN_LAYER_2_1 = config["hidden_layer_2_1"]
        HIDDEN_LAYER_2_2 = config["hidden_layer_2_2"]

        model_config = {
            "feature_size": FEATURE_SIZE,
            "hidden_sizes_1": [HIDDEN_LAYER_1_1, HIDDEN_LAYER_1_2],
            "hidden_sizes_2": [HIDDEN_LAYER_2_1, HIDDEN_LAYER_2_2],
        }

        train_dataloader, val_dataloader, test_dataloader = prepare_dataset(
            pulse_parameters=pulse_parameters,
            unitaries=unitaries,
            expectation_values=expectations,
        )

        lr_scheduler = optax.warmup_cosine_decay_schedule(
            init_value=1e-6,
            peak_value=1e-2,
            warmup_steps=1000,
            decay_steps=10000,
            end_value=1e-6,
        )

        optimiser = optax.adamw(lr_scheduler)

        lr_transformer = default_adaptive_lr_transform(
            PATIENCE=20,
            COOLDOWN=0,
            FACTOR=0.5,
            RTOL=1e-4,
            ACCUMULATION_SIZE=100,
        )

        if not with_dropout:
            model: nn.Module = BasicBlackBox(**model_config)
        else:
            model = MLPBlackBox(**model_config)

        def callback(
            model_params: VariableDict,
            opt_state: optax.OptState,
            history: list[HistoryEntry],
        ) -> None:

            # Check if history[-1].step is divisible by 100
            if (history[-1].global_step + 1) % 100 == 0:
                # Checkpoint the model
                with tempfile.TemporaryDirectory() as tmpdir:
                    model_path = save_model(
                        path=str(tmpdir),
                        experiment_identifier="test",
                        pulse_sequence=pulse_sequence,
                        hamiltonian=rotating_transmon_hamiltonian,
                        model_config=model_config,
                        model_params=model_params,
                        history=history,
                        with_auto_datetime=False,
                    )

                    # Report the loss and val_loss to tune
                    train.report(
                        metrics={
                            "val_loss": history[-1].val_loss,
                            "batch_loss": history[-1].batch_loss,
                        },
                        checkpoint=train.Checkpoint.from_directory(tmpdir),
                    )
            else:
                # Report the loss and val_loss to tune
                train.report(
                    metrics={
                        "val_loss": history[-1].val_loss,
                        "batch_loss": history[-1].batch_loss,
                    },
                )

            return None

        model_opt_params, opt_state, history = train_model(
            key=train_key,
            model=model,
            optimiser=optimiser,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            num_epoch=1500,
            lr_transformer=lr_transformer,
            callback=callback,  # type: ignore
            progress_bar=False,
            loss_type=LossChoice.MSEE,
        )

        return {
            "val_loss": history[-1].val_loss,
            "batch_loss": history[-1].batch_loss,
        }

    return trainable


def default_trainable_v2(
    pulse_sequence: JaxBasedPulseSequence,
    loss_fn: typing.Callable,
):
    def trainable(
        config: dict,
        pulse_parameters: jnp.ndarray,
        unitaries: jnp.ndarray,
        expectations: jnp.ndarray,
        model_key: jnp.ndarray,
        train_key: jnp.ndarray,
    ):

        FEATURE_SIZE = config["feature_size"]
        HIDDEN_LAYER_1_1 = config["hidden_layer_1_1"]
        HIDDEN_LAYER_1_2 = config["hidden_layer_1_2"]
        HIDDEN_LAYER_2_1 = config["hidden_layer_2_1"]
        HIDDEN_LAYER_2_2 = config["hidden_layer_2_2"]

        model_config = {
            "feature_size": FEATURE_SIZE,
            "hidden_sizes_1": [HIDDEN_LAYER_1_1, HIDDEN_LAYER_1_2],
            "hidden_sizes_2": [HIDDEN_LAYER_2_1, HIDDEN_LAYER_2_2],
        }

        train_dataloader, val_dataloader, test_dataloader = prepare_dataset(
            pulse_parameters=pulse_parameters,
            unitaries=unitaries,
            expectation_values=expectations,
        )

        lr_scheduler = optax.warmup_cosine_decay_schedule(
            init_value=1e-6,
            peak_value=1e-2,
            warmup_steps=1000,
            decay_steps=10000,
            end_value=1e-6,
        )

        optimiser = optax.adamw(lr_scheduler)

        model: nn.Module = BasicBlackBox(**model_config)

        def callback(
            model_params: VariableDict,
            opt_state: optax.OptState,
            history: list[HistoryEntryV2],
        ) -> None:

            assert history[-1].loop == "test"

            # Check if history[-1].step is divisible by 100
            if (history[-1].step) % 100 == 0:
                # Checkpoint the model
                with tempfile.TemporaryDirectory() as tmpdir:
                    model_path = save_model(
                        path=str(tmpdir),
                        experiment_identifier="test",
                        pulse_sequence=pulse_sequence,
                        hamiltonian=rotating_transmon_hamiltonian,
                        model_config=model_config,
                        model_params=model_params,
                        history=history,
                        with_auto_datetime=False,
                    )

                    # Report the loss and val_loss to tune
                    train.report(
                        metrics={
                            "test/MSEE": history[-1].MSEE,
                            "test/MAEF": history[-1].MAEF,
                        },
                        checkpoint=train.Checkpoint.from_directory(tmpdir),
                    )
            else:
                # Report the loss and val_loss to tune
                train.report(
                    metrics={
                        "test/MSEE": history[-1].MSEE,
                        "test/MAEF": history[-1].MAEF,
                    },
                )

            return None

        model_params, opt_state, history = train_model_v2(
            train_dataloader=train_dataloader,
            validation_data_loader=val_dataloader,
            test_dataloader=test_dataloader if test_dataloader is not None else [],
            model=model,
            optimizer=optimiser,
            loss_fn=loss_fn,
            model_init_key=model_key,
            NUM_EPOCHS=1500,
            callbacks=[callback],
        )

        return {
            "test/MSEE": history[-1].MSEE,
            "test/MAEF": history[-1].MAEF,
        }

    return trainable


class SearchAlgo(Enum):
    HYPEROPT = "hyperopt"
    OPTUNA = "optuna"


def hypertuner(
    trainable: typing.Callable,
    pulse_parameters: jnp.ndarray,
    unitaries: jnp.ndarray,
    expectations: jnp.ndarray,
    model_key: jnp.ndarray,
    train_key: jnp.ndarray,
    num_samples: int = 100,
    search_algo: SearchAlgo = SearchAlgo.HYPEROPT,
    metric: str = "val_loss",
):

    search_space = {
        "feature_size": tune.randint(5, 10),
        "hidden_layer_1_1": tune.randint(5, 50),
        "hidden_layer_1_2": tune.randint(5, 50),
        "hidden_layer_2_1": tune.randint(5, 50),
        "hidden_layer_2_2": tune.randint(5, 50),
    }

    current_best_params = [
        {
            "feature_size": 5,
            "hidden_layer_1_1": 10,
            "hidden_layer_1_2": 20,
            "hidden_layer_2_1": 10,
            "hidden_layer_2_2": 20,
        }
    ]

    if search_algo == SearchAlgo.HYPEROPT:
        search_algo_instance: Searcher = HyperOptSearch(
            metric=metric, mode="min", points_to_evaluate=current_best_params
        )
    elif search_algo == SearchAlgo.OPTUNA:
        search_algo_instance = OptunaSearch(metric=metric, mode="min")

    run_config = train.RunConfig(
        name="tune_experiment",
        checkpoint_config=train.CheckpointConfig(
            num_to_keep=10,
        ),
    )

    tuner = tune.Tuner(
        tune.with_parameters(
            trainable,
            pulse_parameters=pulse_parameters,
            unitaries=unitaries,
            expectations=expectations,
            model_key=model_key,
            train_key=train_key,
        ),
        tune_config=tune.TuneConfig(
            search_alg=search_algo_instance,
            metric=metric,
            mode="min",
            num_samples=num_samples,
        ),
        param_space=search_space,
        run_config=run_config,
    )

    results = tuner.fit()

    return results


def get_best_hypertuner_results(results, metric: str = "val_loss"):
    with results.get_best_result(
        metric=metric, mode="min"
    ).checkpoint.as_directory() as checkpoint_dir:
        model_state, hist, data_config = load_model(str(checkpoint_dir))
    return model_state, hist, data_config
