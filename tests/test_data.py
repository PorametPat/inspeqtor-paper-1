import pytest
import inspeqtor as isq
import jax.numpy as jnp
import jax
import pandas as pd  # type: ignore

qubit_test_cases = [
    {
        "qubit_info": isq.data.QubitInformation(
            unit="GHz",
            qubit_idx=0,
            anharmonicity=0.2,
            frequency=5.0,
            drive_strength=0.1,
        ),
    },
    {
        "qubit_info": isq.data.QubitInformation(
            unit="Hz", qubit_idx=0, anharmonicity=0.2, frequency=5.0, drive_strength=0.1
        ),
    },
    {
        "qubit_info": isq.data.QubitInformation(
            unit="2piGHz",
            qubit_idx=0,
            anharmonicity=0.2 * 2 * jnp.pi,
            frequency=5.0 * 2 * jnp.pi,
            drive_strength=0.1 * 2 * jnp.pi,
        ),
    },
    {
        "qubit_info": isq.data.QubitInformation(
            unit="2piHz",
            qubit_idx=0,
            anharmonicity=-2118283466.642022,
            frequency=33030879605.3489,
            drive_strength=952346606.9510472,
        ),
    },
]


@pytest.mark.parametrize("state_str", ["0", "1", "+", "-", "r", "l"])
def test_State(state_str):
    state = isq.data.State.from_label(state_str)

    assert state.shape == (2, 1)

    rho = isq.data.State.from_label(state_str, dm=True)

    assert rho.shape == (2, 2)

    assert jnp.allclose(rho, jnp.outer(state, state.T.conj()))

    qutrit_rho = isq.data.State.to_qutrit(rho)

    assert qutrit_rho.shape == (3, 3)


@pytest.mark.parametrize("test_case", qubit_test_cases)
def test_QubitInformation(test_case):
    qubit_info = test_case["qubit_info"]

    # Check that the unit is converted to GHz
    assert qubit_info.unit == "GHz"

    dict_qubit_info = qubit_info.to_dict()

    # From dict to dataclass
    qubit_info_from_dict = isq.data.QubitInformation.from_dict(dict_qubit_info)

    assert qubit_info == qubit_info_from_dict


def test_expectation_value():

    expectation_value = isq.data.ExpectationValue(observable="X", initial_state="+")

    assert expectation_value == isq.data.ExpectationValue(
        observable="X", initial_state="+"
    )
    assert expectation_value != isq.data.ExpectationValue(
        observable="X", initial_state="-"
    )

    # Serialization and deserialization
    expectation_value_dict = expectation_value.to_dict()

    expectation_value_from_dict = isq.data.ExpectationValue.from_dict(
        expectation_value_dict
    )

    assert expectation_value == expectation_value_from_dict


def test_ExperimentConfigV3(tmp_path):

    qubit_info = qubit_test_cases[0]["qubit_info"]

    experiment_config = isq.data.ExperimentConfiguration(
        qubits=[qubit_info],
        expectation_values_order=isq.utils.predefined.default_expectation_values_order,
        parameter_names=[["param1", "param2"], ["param3", "param4"]],
        backend_name="qasm_simulator",
        sample_size=2,
        shots=1000,
        EXPERIMENT_IDENTIFIER="0001",
        EXPERIMENT_TAGS=["test", "test2"],
        description="This is a test experiment",
        device_cycle_time_ns=2 / 9,
        sequence_duration_dt=10,
        instance="open",
    )

    # To dict
    dict_experiment_config = experiment_config.to_dict()

    # From dict to dataclass
    experiment_config_from_dict = isq.data.ExperimentConfiguration.from_dict(
        dict_experiment_config
    )

    assert experiment_config == experiment_config_from_dict

    d = tmp_path / "test"
    d.mkdir()

    # Test to_file()
    experiment_config.to_file(path=str(d))

    # Test from_file()
    experiment_config_from_file = isq.data.ExperimentConfiguration.from_file(
        path=f"{str(d)}"
    )

    assert experiment_config == experiment_config_from_file


def test_get_parameters_dict_list_1():
    parameters_name = [
        ["a", "b"],
        ["c", "d"],
    ]
    parameters_row = pd.Series(
        {
            "parameter/0/a": 1.0,
            "parameter/0/b": 2.0,
            "parameter/1/c": 3.0,
            "parameter/1/d": 4.0,
            "parameter/2/e": 5.0,
        }
    )

    assert isq.data.get_parameters_dict_list(parameters_name, parameters_row) == [
        {"a": 1.0, "b": 2.0},
        {"c": 3.0, "d": 4.0},
    ]


def test_get_parameters_dict_list_2():
    key = jax.random.PRNGKey(0)
    pulse_sequence = isq.utils.predefined.get_multi_drag_pulse_sequence_v2()
    pulse_params = pulse_sequence.sample_params(key)
    parameters_name = pulse_sequence.get_parameter_names()
    # Flatten the pulse_params
    parameters_dict = {}
    for i, params in enumerate(pulse_params):
        for k, v in params.items():
            parameters_dict[f"parameter/{i}/{k}"] = float(v)

    parameters_row = [parameters_dict]
    df_params = pd.DataFrame(parameters_row)

    parameters_dict_list = isq.data.get_parameters_dict_list(
        parameters_name, df_params.iloc[0]
    )

    assert parameters_dict_list == pulse_params


def test_ExperimentData_1(tmp_path):
    qubit_info, pulse_sequence, config = isq.utils.predefined.get_mock_prefined_exp_v1()
    key = jax.random.PRNGKey(0)

    # Generate mock expectation value
    key, exp_key = jax.random.split(key)
    expectation_values = 2 * (jax.random.uniform(exp_key, shape=(10, 18)) - (1 / 2))

    rows = []
    manual_pulse_params_array = []
    pulse_params_list = []
    parameter_structure = pulse_sequence.get_parameter_names()
    for sample_idx in range(config.sample_size):

        key, subkey = jax.random.split(key)
        pulse_params = pulse_sequence.sample_params(subkey)
        pulse_params_list.append(pulse_params)
        manual_pulse_params_array.append(
            isq.pulse.list_of_params_to_array(pulse_params, parameter_structure)
        )

        for exp_idx, exp in enumerate(
            isq.utils.predefined.default_expectation_values_order
        ):
            row = isq.data.make_row(
                expectation_value=float(expectation_values[sample_idx, exp_idx]),
                initial_state=exp.initial_state,
                observable=exp.observable,
                parameters_list=pulse_params,
                parameters_id=sample_idx,
            )

            rows.append(row)

    df = pd.DataFrame(rows)

    manual_pulse_params_array = jnp.array(manual_pulse_params_array)
    exp_data = isq.data.ExperimentData(experiment_config=config, preprocess_data=df)

    # Check that the parameter extract from dataframe is the same as the one that created from pulse sequence directly.
    assert jnp.allclose(exp_data.parameters, manual_pulse_params_array)

    # Check that the list[ParameterDictType] has the same structure and value upto 10 decimal.
    for exp_data_param, pulse_param in zip(
        exp_data.get_parameters_dict_list(), pulse_params_list
    ):
        assert exp_data_param == jax.tree.map(
            lambda x: round(x, exp_data.keep_decimal), pulse_param
        )

    # Check the expectation values
    expectation_values = exp_data.get_expectation_values()
    assert expectation_values.shape == (config.sample_size, 18)
    assert jnp.allclose(expectation_values, expectation_values)

    # Test the save and load
    d = tmp_path / "test"
    d.mkdir()

    exp_data.save_to_folder(path=str(d))

    exp_data_from_file = isq.data.ExperimentData.from_folder(path=str(d))

    assert exp_data == exp_data_from_file
