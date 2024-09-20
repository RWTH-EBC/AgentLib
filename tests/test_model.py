"""Module with tests for all models in the agentlib."""

import os
import sys
import json
import unittest
from pydantic import ValidationError
import numpy as np
from agentlib.core import Model
from agentlib.models.scipy_model import ScipyStateSpaceModel
from agentlib.models.fmu_model import FmuModel
from agentlib.core import datamodels


class DummyModel(Model):
    def do_step(self, *, t_start, t_sample=None):
        pass

    def initialize(self, **kwargs):
        pass


class TestModel(unittest.TestCase):
    """Class to test only the base-model."""

    def setUp(self) -> None:
        self.name = "Test"
        self.names = [self.name]
        self.inp = datamodels.ModelInput(name=self.name)
        self.out = datamodels.ModelOutput(name=self.name)
        self.par = datamodels.ModelParameter(name=self.name, value=10)
        self.sta = datamodels.ModelState(name=self.name, value=1)

        self.model = DummyModel(
            inputs=[{"name": self.name}],
            outputs=[{"name": self.name}],
            states=[{"name": self.name, "value": 1}],
            parameters=[{"name": self.name, "value": 10}],
        )

    def test_description(self):
        """Test if the description is set and deleted."""
        _default_desc = self.model.get_config_type().model_fields["description"].default
        _test_desc = "This is just a dummy description"
        self.model.description = _test_desc
        self.assertEqual(_test_desc, self.model.description)
        del self.model.description
        self.assertEqual(_default_desc, self.model.description)
        with self.assertRaises(ValidationError):
            self.model.description = [397]

    def test_properties(self):
        """Test model properties"""
        model = DummyModel()
        random_number = np.random.randint(100)
        model.sim_time = random_number
        self.assertEqual(model.sim_time, random_number)
        del model.sim_time
        self.assertIsInstance(model.sim_time, float)
        self.assertIsInstance(model.dt, (float, int))
        model.name = f"SomeName_{random_number}"
        self.assertEqual(model.name, f"SomeName_{random_number}")
        del model.name
        self.assertIsInstance(model.name, str)
        self.assertEqual(model.variables, [])

    def test_reject_non_value(self):
        """Tests, that the model setter rejects a None value"""
        self.model.set("Test", 10)
        self.model.set("Test", None)
        self.assertEqual(self.model.get("Test").value, 10)

    def test_variables(self):
        """Test if variables are accessible"""
        # Test getting by name
        self.assertEqual(self.inp, self.model.get_input(self.name))
        self.assertEqual(self.inp, self.model.get_inputs(self.names)[0])
        self.assertEqual(self.out, self.model.get_output(self.name))
        self.assertEqual(self.out, self.model.get_outputs(self.names)[0])
        self.assertEqual(self.sta, self.model.get_state(self.name))
        self.assertEqual(self.sta, self.model.get_states(self.names)[0])
        self.assertEqual(self.par, self.model.get_parameter(self.name))
        self.assertEqual(self.par, self.model.get_parameters(self.names)[0])

    def test_abc(self):
        """Test if abc evokes error"""
        with self.assertRaises(TypeError):
            model = Model()

    def test_get_attr(self):
        """Test if AttributeError is evoked"""
        model = DummyModel()
        with self.assertRaises(AttributeError):
            model.not_a_valid_var_name

    def test_generate_config(self):
        """Test the config generator"""
        model = DummyModel()
        filename = model.generate_variables_config()
        with open(filename, "r") as file:
            config = json.load(file)
            self.assertIsInstance(config, dict)
            self.assertTrue("inputs" in config.keys())
            self.assertTrue("states" in config.keys())
            self.assertTrue("outputs" in config.keys())
            self.assertTrue("parameters" in config.keys())

    def test_get_set(self):
        random_val = np.random.randint(100)
        model = DummyModel(
            inputs=[{"name": "test_inp", "value": 0}],
            outputs=[{"name": "test_out", "value": 0}],
            parameters=[{"name": "test_par", "value": 0}],
            states=[{"name": "test_sta", "value": 0}],
        )
        model.set("test_inp", random_val)
        model.set("test_out", random_val)
        model.set("test_par", random_val)
        model.set("test_sta", random_val)
        model.set_parameter_value(name="test_par", value=random_val)
        model._set_state_value(name="test_sta", value=random_val)
        model.set_input_value(name="test_inp", value=random_val)
        model._set_output_value(name="test_out", value=random_val)

        self.assertEqual(model.get("test_inp").value, random_val)
        self.assertEqual(model.get("test_out").value, random_val)
        self.assertEqual(model.get("test_par").value, random_val)
        self.assertEqual(model.get("test_sta").value, random_val)
        with self.assertRaises(ValueError):
            model.get("not_an_input")
        with self.assertRaises(ValueError):
            model.set("not_an_input", 10)

    def test_create_time_samples(self):
        """Test create time samples"""
        model = DummyModel()
        random_val = np.random.randint(10, 100)
        samples = model._create_time_samples(t_sample=random_val)
        self.assertEqual(samples[-1], random_val)
        samples = model._create_time_samples(t_sample=random_val + 0.01)
        self.assertEqual(samples[-1], random_val + 0.01)


class TestFMUModel(unittest.TestCase):
    """Class to test the FMUModel"""

    def setUp(self) -> None:
        self.path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "examples",
            "models",
            "fmu",
            "test_sinus.fmu",
        )
        if "linux" in sys.platform:
            self.skipTest("FMU test not yet supported")
        self.model = FmuModel(
            path=self.path, only_config_variables=False, dt=np.random.rand() + 0.1
        )

    def test_initialize(self):
        """Test initialization of FMU"""
        self.model.initialize(t_start=0, t_stop=10)

    def test_simulate(self) -> None:
        """Test simulation of FMU"""
        self.model.initialize(t_start=0, t_stop=3600)

        self.model.set_input_values(names=["u_mul", "u_add"], values=[5, -5])
        self.model.do_step(t_start=0, t_sample=3600)


class TestScipyModel(unittest.TestCase):
    """Class to test the ScipyModel"""

    def setUp(self) -> None:
        param_a = np.random.rand() + 1
        param_b = np.random.rand() + 1

        self.inputs = [
            datamodels.ModelInput(name="T_oda", value=273.15),
            datamodels.ModelInput(name="Q_flow_heat", value=0),
        ]
        self.out = datamodels.ModelOutput(name="T_room")
        self.sta = datamodels.ModelState(name="T_room", value=293.15)
        self.model = ScipyStateSpaceModel(
            system={
                "A": [-param_a / param_b],
                "B": [param_a / param_b, 1 / param_b],
                "C": [1],
                "D": [0, 0],
            },
            inputs=[
                {"name": "T_oda", "value": 273.15},
                {"name": "Q_flow_heat", "value": 0},
            ],
            outputs=[{"name": "T_room"}],
            states=[{"name": "T_room", "value": 293.15}],
            dt=np.random.rand() + 0.1,
        )

    def test_simulation(self):
        """Test the simulation of the scipy model"""
        self.model.do_step(t_start=0, t_sample=3600)
        # System should be in equilibrium
        self.assertEqual(
            round(self.model.get_output("T_room").value, 2),
            self.model.get_input("T_oda").value,
        )


if __name__ == "__main__":
    unittest.main()
