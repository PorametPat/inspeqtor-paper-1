import jax
import inspeqtor as sq
import optax # type: ignore

def test_training_loop_v2():

    key = jax.random.key(0)
    exp_data, pulse_sequence, noisy_unitaries, signal_parameters_list, noisy_simulator = (
        sq.utils.helper.generate_mock_experiment_data(
            key=key, sample_size=1000, strategy=sq.utils.helper.SimulationStrategy.IDEAL
        )
    )

    loaded_data = sq.utils.helper.prepare_data(
        exp_data=exp_data, pulse_sequence=pulse_sequence
    )

    train_dataloader, validation_data_loader, test_dataloader = sq.data.prepare_dataset(
        pulse_parameters=loaded_data.pulse_parameters,
        unitaries=loaded_data.unitaries,
        expectation_values=loaded_data.expectation_values
    )

    model_init_key, key = jax.random.split(key)

    model_params, opt_state, history = sq.model_v2.train_model(
        train_dataloader=train_dataloader,
        validation_data_loader=validation_data_loader,
        test_dataloader=test_dataloader,
        model=sq.model.BasicBlackBox(feature_size=5),
        optimizer=optax.adam(1e-3),
        loss_fn=sq.model_v2.MAEF_loss_fn,
        model_init_key=model_init_key,
        NUM_EPOCHS=10,
    )