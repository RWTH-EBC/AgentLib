"""Module to test all datamodels in the agentlib"""

import unittest
from pydantic import ValidationError
from agentlib.core import datamodels, errors
import pandas as pd


class TestVariables(unittest.TestCase):
    """Class with tests for Variables"""

    def setUp(self) -> None:
        self.config = {
            "name": "testing",
            "type": None,
            "value": None,
            "unit": "-",
            "description": "Description of the variable",
        }

    def test_base_model_variable(self):
        # Default variability
        var = datamodels.BaseModelVariable(
            name="test", causality=datamodels.Causality.input
        )
        self.assertEqual(var.variability, datamodels.Variability.continuous)
        var = datamodels.BaseModelVariable(
            name="test", causality=datamodels.Causality.parameter
        )
        self.assertEqual(var.variability, datamodels.Variability.tunable)

    def test_value_conversion(self):
        """Test value conversion when providing a type."""
        var = datamodels.BaseVariable.validate_data(
            dict(name="test", value=100.0, type="float")
        )
        var.set_value("100", validate=True)
        self.assertEqual(var.value, 100)
        self.assertIsInstance(var.value, float)
        with self.assertRaises(ValueError):
            var.set_value("not a float", validate=True)

    def test_series_sending(self):
        """Test pd.Series are sent and received correctly."""
        srs = pd.Series([0, 1, 2, 3])

        # emulates sending and receiving
        var = datamodels.AgentVariable.validate_data(
            dict(name="test", value=srs, type="pd.Series")
        )
        json = var.json()
        var = datamodels.AgentVariable.from_json(json)

        self.assertIsInstance(var.value, pd.Series)
        self.assertIsInstance(var.value.index[0], (float, int))

    def test_type(self):
        """Test the type setting"""
        with self.assertRaises(ValueError):
            datamodels.BaseVariable.validate_data(
                dict(name="test", type="not_supported")
            )
        with self.assertRaises(TypeError):
            datamodels.BaseVariable.validate_data(dict(name="test", type=float))

    def test_allowed_values(self):
        """Test allowed values"""
        with self.assertRaises(ValueError):
            datamodels.BaseVariable.validate_data(
                dict(
                    name="test",
                    type="float",
                    ub=10,
                    lb=0,
                    clip=True,
                    allowed_values=[20],
                )
            )
        var = datamodels.BaseVariable.validate_data(
            dict(
                name="test",
                type="float",
                value=20.0,
                allowed_values=["0", "10", 20, 30.0],
            )
        )
        for v in var.allowed_values:
            self.assertEqual(type(v), float)
        with self.assertRaises(ValueError):
            var.set_value(14.5, validate=True)  # Not in list.

    def test_pd_series(self):
        """Test the special cases for pd.Series"""
        srs = pd.Series([0, 100], index=[0, 10])
        srs_json = srs.to_json()
        srs_literal = srs.to_dict()
        for value in [srs_json, srs_literal]:
            var = datamodels.BaseVariable.validate_data(
                dict(
                    name="test",
                    type="pd.Series",
                    value=value,
                    ub=10,
                    lb=5,
                    clip=True,
                    allowed_values=[20],
                )
            )
            self.assertEqual(var.allowed_values, [])
            self.assertIsInstance(var.value, pd.Series)

    def test_bounds(self):
        """Test if ub and lb work"""
        var = datamodels.BaseVariable.validate_data({"name": "test"})
        # Test correct setting
        var.value = 100  # Set variable
        var.lb = 0
        var.ub = 200
        # Test wrong setting:
        with self.assertRaises(ValueError):
            datamodels.BaseVariable.validate_data(dict(name="test", ub=100, lb=200))

        # With clip=True:
        var = datamodels.BaseVariable.validate_data(
            dict(name="test", clip=True, ub=100, lb=0, value=10000)
        )
        self.assertEqual(var.value, var.ub)
        var = datamodels.BaseVariable.validate_data(
            dict(name="test", clip=True, ub=100, lb=0, value=-10000)
        )
        self.assertEqual(var.value, var.lb)

    def test_agent_variables(self):
        """Test if AgentVariables work"""
        var = datamodels.AgentVariable.validate_data(self.config)
        # Test alias
        self.assertEqual(var.alias, var.name)
        # Test custom alias:
        custom_alias = "MyCustomAlias"
        config = self.config.copy()
        config.update({"alias": custom_alias})
        internal = datamodels.AgentVariable.validate_data(config)
        self.assertEqual(internal.alias, custom_alias)

    def test_source_parsing(self):
        """Test if source parsing works"""
        ag_id = "MyAgTests"
        mod_id = "MyModTest"
        var = datamodels.AgentVariable(name="test", source=ag_id)
        self.assertEqual(var.source.agent_id, ag_id)
        var = datamodels.AgentVariable(
            name="test", source={"agent_id": ag_id, "module_id": mod_id}
        )
        self.assertEqual(var.source.agent_id, ag_id)
        self.assertEqual(var.source.module_id, mod_id)
        var = datamodels.AgentVariable(
            name="test", source=datamodels.Source(agent_id=ag_id, module_id=mod_id)
        )
        self.assertEqual(var.source.agent_id, ag_id)
        self.assertEqual(var.source.module_id, mod_id)

    def test_source_matching(self):
        """Test if source matching works"""
        s_base = datamodels.Source(agent_id="Test", module_id="Test_2")
        s_2 = datamodels.Source(agent_id="Test", module_id="Test_2")
        self.assertTrue(s_base.matches(s_2))
        # Test if None is specified:
        s_2 = datamodels.Source(agent_id="Test")
        self.assertTrue(s_2.matches(s_base))
        s_2 = datamodels.Source(module_id="Test_2")
        self.assertTrue(s_2.matches(s_base))
        self.assertTrue(datamodels.Source().matches(s_base))

    def test_model_variables(self):
        """Test if ModelVariables work"""
        inp = datamodels.ModelInput(**self.config)
        self.assertEqual(inp.causality, datamodels.Causality.input)
        out = datamodels.ModelOutput(**self.config)
        self.assertEqual(out.causality, datamodels.Causality.output)
        par = datamodels.ModelParameter(
            **{
                **self.config,
                "value": 10,
            }
        )
        self.assertEqual(par.causality, datamodels.Causality.parameter)
        sta = datamodels.ModelState(
            **{
                **self.config,
                "value": 10,
            }
        )
        self.assertEqual(sta.causality, datamodels.Causality.local)

    def test_calculation(self):
        """Test model calculations"""
        inp = datamodels.ModelInput(name="test", value=1)
        self.assertEqual(inp + 5, 1 + 5)
        self.assertEqual(inp * 5, 1 * 5)
        self.assertEqual(inp - 5, 1 - 5)
        self.assertEqual(inp / 5, 1 / 5)
        self.assertEqual(inp**5, 1**5)

        self.assertEqual(5 + inp, 5 + 1)
        self.assertEqual(5 * inp, 5 * 1)
        self.assertEqual(5 - inp, 5 - 1)
        self.assertEqual(5 / inp, 5 / 1)
        self.assertEqual(5**inp, 5**1)

    def test_equal(self):
        """Test agent equal"""
        var = datamodels.AgentVariable(name="test")
        self.assertEqual(var, var.copy())

    def test_source(self):
        source = datamodels.Source(agent_id="test", module_id="test2")
        self.assertIsInstance(source.__str__(), str)
        self.assertIsInstance(hash(source), type(hash(1)))
        self.assertEqual(source, source)
        self.assertEqual(source, source.dict())
        self.assertNotEqual(source, 5)

    def test_serializing(self):
        """Test if Serialization works, especially with Series."""
        conf = self.config

        # test on normal value
        conf["value"] = 12
        var = datamodels.AgentVariable(**conf)
        json_str = var.json()
        var_after = datamodels.AgentVariable.from_json(json_str)
        self.assertEqual(var, var_after)


if __name__ == "__main__":
    unittest.main()
