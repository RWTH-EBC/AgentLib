"""Module with tests for the core module of the agentlib."""

import itertools
import logging
import os
import unittest
from typing import Optional, List

from pydantic import ValidationError

from agentlib.core import (
    BaseModule,
    Agent,
    Environment,
    datamodels,
    BaseModuleConfig,
    AgentVariables,
    AgentVariable,
)
from agentlib.core.errors import ConfigurationError


default_data = {
    "type": "float",
    "value": 100,
    "ub": 200,
    "lb": 0,
    "allowed_values": [100, 150],
    "shared": True,
    "unit": "testUnit",
    "description": "My Doc",
    "clip": True,
    "rdf_class": "MyRDFClass",
    "source": {"agent_id": "2938310293"},
}


LOG_LEVELS = ["INFO", "DEBUG", "CRITICAL", "WARNING", "ERROR"]


class CustomModuleConfig(BaseModuleConfig):
    my_shared_field: AgentVariables = []
    my_single_agent_var: AgentVariable = AgentVariable(name="u1", **default_data)
    my_single_agent_vars: AgentVariables = [AgentVariable(name="u2", **default_data)]
    my_custom_vars: AgentVariables = []
    shared_variable_fields: List[str] = ["my_shared_field"]


class CustomModule(BaseModule):
    config: CustomModuleConfig

    def register_callbacks(self):
        pass

    def process(self):
        yield self.env.event()


class BrokenCustomModule(BaseModule):
    def register_callbacks(self):
        pass

    def process(self):
        yield self.env.event()


class TestModuleConfig(unittest.TestCase):
    """Class to test only the module settings."""

    def setUp(self) -> None:
        """Setup the test-logger."""
        self.mod_id = "test_mod_id"
        self.ag_id = "test_ag_id"
        self.test_config = {
            "type": "not_necessary_here",
            "module_id": self.mod_id,
        }

    def test_source(self):
        """Test whether sources are correctly added."""
        mod = CustomModuleConfig(_agent_id=self.ag_id, **self.test_config)
        self.assertEqual(self.mod_id, mod.module_id)
        for var in mod.get_variables():
            self.assertIsNotNone(var.source.agent_id)
            self.assertIsNone(var.source.module_id)

    def test_duplicate_names(self):
        """Tests if duplicate names raises an error."""
        cfg = {
            "my_single_agent_vars": [
                {"name": "Test0"},
                {"name": "Test0"},
            ],
        }
        with self.assertRaises(ValueError):
            CustomModuleConfig(_agent_id=self.ag_id, **self.test_config, **cfg)

    def test_make_shared(self):
        """Tests if shared fields are properly reconfigured"""
        cfg = {"my_shared_field": [{"name": "test"}]}
        my_config = CustomModuleConfig(_agent_id=self.ag_id, **self.test_config, **cfg)
        self.assertTrue(my_config.my_shared_field[0].shared)

    def test_make_shared_raises_on_misspelling(self):
        """Tests if ConfigurationError is raised when providing wrong shared fields"""
        cfg = {"shared_variable_fields": ["non_existing_field"]}
        with self.assertRaises(ConfigurationError):
            CustomModuleConfig(_agent_id=self.ag_id, **self.test_config, **cfg)

    def test_overwrite_default(self):
        """Test the overwrite default behaviour"""
        self._check_if_overwrite_works(
            overwrite_default={
                "type": "int",
                "value": 200,
                "ub": 400,
                "lb": -200,
                "allowed_values": [100, 150, 200],
                "shared": False,
                "unit": "testUnit2",
                "description": "My Doc 2",
                "clip": False,
                "rdf_class": "MyRDFClass2",
                "source": {"agent_id": "2938310293", "module_id": None},
            }
        )
        self._check_if_overwrite_works(overwrite_default={"type": "int"})

    def _check_if_overwrite_works(self, overwrite_default: dict):
        """Helper to avoid boiler plate and test various custom AgentVars"""
        cfg = {
            "my_single_agent_var": {"name": "u1", **overwrite_default},
            "my_single_agent_vars": [{"name": "u2", **overwrite_default}],
            "my_custom_vars": [{"name": "u3", **overwrite_default}],
        }
        updated_data = default_data.copy()
        updated_data.update(overwrite_default)
        my_config = CustomModuleConfig(_agent_id=self.ag_id, **self.test_config, **cfg)
        for key, value in updated_data.items():
            if key == "source":
                shared = overwrite_default.get("shared")
                self._check_source(
                    shared=shared,
                    to_check=my_config.my_single_agent_var.dict()[key],
                    truth=value,
                )
                self._check_source(
                    shared=shared,
                    to_check=my_config.my_single_agent_vars[0].dict()[key],
                    truth=value,
                )
                continue

            self.assertEqual(my_config.my_single_agent_var.dict()[key], value)
            self.assertEqual(my_config.my_single_agent_vars[0].dict()[key], value)

    def _check_source(self, shared: Optional[bool], truth: dict, to_check: dict):
        shared_fields = CustomModuleConfig.model_fields[
            "shared_variable_fields"
        ].default
        shared_source = {"agent_id": self.ag_id, "module_id": None}

        if shared:
            # case where user config makes this shared
            self.assertEqual(to_check, shared_source)
            return
        elif shared is not None:
            # case where user config makes this not shared
            self.assertEqual(to_check, truth)
        elif to_check in shared_fields:
            # case where the variable is in shared fields and user did not specify
            self.assertEqual(to_check, shared_source)
        elif default_data["shared"]:
            # case where the variable is not in shared fields and user did not specify
            # but default has shared
            self.assertEqual(to_check, shared_source)
        else:
            # case where the variable is not in shared fields and user did not specify
            # and default has not shared
            self.assertEqual(to_check, truth)

    def test_user_config(self):
        """Test the overwrite default behaviour"""
        cfg = {}
        user_config = {**self.test_config, **cfg}
        my_config = CustomModuleConfig(_agent_id=self.ag_id, **user_config)
        self.assertEqual(my_config._user_config, user_config)

    def test_log_level(self):
        """Test log_level settings"""
        # Test all
        for lvl in LOG_LEVELS:
            BaseModuleConfig(_agent_id=self.ag_id, log_level=lvl, **self.test_config)
        # Test strs
        BaseModuleConfig(_agent_id=self.ag_id, log_level="InFo", **self.test_config)
        BaseModuleConfig(_agent_id=self.ag_id, log_level="warning", **self.test_config)
        BaseModuleConfig(_agent_id=self.ag_id, log_level="criTical", **self.test_config)
        BaseModuleConfig(_agent_id=self.ag_id, log_level=None, **self.test_config)
        with self.assertRaises(ValidationError):
            BaseModuleConfig(
                _agent_id=self.ag_id, log_level="criticall", **self.test_config
            )
        with self.assertRaises(ValidationError):
            BaseModuleConfig(
                _agent_id=self.ag_id, log_level="infoo", **self.test_config
            )

    def test_extra_error_message(self):
        """Checks that the error message is correctly modified with suggestions when
        initializing with wrong field names"""
        cfg = {
            "extra-field": "ignored",
        }
        with self.assertRaises(ValidationError):
            _ = BaseModuleConfig(_agent_id=self.ag_id, **self.test_config, **cfg)
        try:
            _ = BaseModuleConfig(_agent_id=self.ag_id, **self.test_config, **cfg)
        except ValidationError as e:
            suggestions = ["module_id", "type", "log_level"]
            for suggestion in suggestions:
                self.assertIn(suggestion, str(e))

    def test_custom_json_schema(self):
        """Checks that the json schema is implemented correctly, putting log level and
        shared variable fields always last."""
        schema = CustomModuleConfig.model_json_schema()
        fields = list(schema["properties"])
        self.assertEqual(fields[-1], "log_level")
        self.assertEqual(fields[-2], "shared_variable_fields")

    def test_default(self):
        """Checks implementation of the default utility method"""
        self.assertEqual(
            CustomModuleConfig.default("my_single_agent_var"),
            AgentVariable(name="u1", **default_data),
        )


class TestModule(unittest.TestCase):
    """Class to test only the module settings."""

    def setUp(self) -> None:
        """Setup the test-logger."""
        self.mod_id = "test_mod_id"
        self.ag_id = "test_ag_id"
        self.test_config = {
            "type": "not_necessary_here",
            "module_id": self.mod_id,
            "my_custom_vars": [{"name": "test_inp"}],
        }
        self.agent = Agent(config={"id": "Test", "modules": []}, env=Environment())

    def test_missing_config_type(self):
        """Tests if a proper error is raised when the type is missing."""
        with self.assertRaises(ConfigurationError):
            mod = BrokenCustomModule(config=self.test_config, agent=self.agent)

    def test_properties(self):
        """Test properties of module"""
        module = CustomModule(config=self.test_config, agent=self.agent)
        self.assertEqual(len(module.variables), 3)

        # make sure copy is returned
        variables = module.variables
        orig_variables = list(module._variables_dict.values())
        self.assertIsNot(variables[0], orig_variables[0])
        self.assertEqual(variables[0], orig_variables[0])

    def test_get_set(self):
        """Tests the get and set methods."""
        module = CustomModule(config=self.test_config, agent=self.agent)

        # test non existing variable
        with self.assertRaises(KeyError):
            module.get("Not an input")

        # test existing variable
        test_out = module.get("test_inp")
        self.assertIsInstance(test_out, datamodels.AgentVariable)

        # make sure copy is returned
        self.assertIsNot(test_out, module._variables_dict["test_inp"])
        self.assertEqual(test_out, module._variables_dict["test_inp"])

        # check setting
        test_out.value = 42
        test_out.timestamp = self.agent.env.time
        module.set("test_inp", 42)
        self.assertIsNot(test_out, module._variables_dict["test_inp"])
        self.assertEqual(test_out, module._variables_dict["test_inp"])
        self.assertEqual(module._variables_dict["test_inp"].value, 42)

        with self.assertRaises(KeyError):
            module.set("not a variable", 43)

    def test_get_value(self):
        """Tests the get_value method."""
        module = CustomModule(config=self.test_config, agent=self.agent)
        # make value mutable
        module.set("test_inp", {"a": "b"})
        test_value = module.get_value("test_inp")

        # make sure copy is returned
        self.assertIsNot(test_value, module._variables_dict["test_inp"].value)
        self.assertEqual(test_value, module._variables_dict["test_inp"].value)

    def test_update_variables(self):
        """Test get, update and set methods"""
        module = CustomModule(config=self.test_config, agent=self.agent)

        # test updating a single variable
        module.update_variables([datamodels.AgentVariable(name="test_inp")])

        # test updating multiple variables
        module.update_variables(
            [
                datamodels.AgentVariable(name="test_inp"),
                datamodels.AgentVariable(name="test_inp"),
            ]
        )

        # test updating a variable which is not in the config
        with self.assertRaises(ValueError):
            module.update_variables(
                [
                    datamodels.AgentVariable(
                        name="Not in config, however " "does not raise any error."
                    )
                ]
            )

    def test_timestamp_setting(self):
        """Test the setting of timestamps"""
        module = CustomModule(config=self.test_config, agent=self.agent)
        module.set("test_inp", value=10, timestamp=10)
        self.assertEqual(module.get("test_inp").timestamp, 10)
        module.set("test_inp", value=10)
        self.assertEqual(module.get("test_inp").timestamp, 0)
        module.update_variables(
            [datamodels.AgentVariable(name="test_inp")], timestamp=10
        )
        self.assertEqual(module.get("test_inp").timestamp, 10)


class TestModuleLogging(unittest.TestCase):
    def setUp(self) -> None:
        self.test_config = {
            "type": "not_necessary_here",
            "module_id": "test_mod_id",
        }
        self.agent = Agent(config={"id": "Test", "modules": []}, env=Environment())
        self.log_filename = "log_test.log"

    def tearDown(self) -> None:
        for h in logging.getLogger().handlers:
            logging.getLogger().removeHandler(h)
            h.close()
        try:
            os.remove(self.log_filename)
        except (PermissionError, FileNotFoundError) as err:
            print(err)

    def test_log_levels(self):
        """Test the log_level options"""
        for root_lvl, mod_lvl in itertools.product(LOG_LEVELS, LOG_LEVELS):
            logging.basicConfig(level=root_lvl, filename=self.log_filename)
            mod_lvl_int = logging.getLevelName(mod_lvl)
            self._clear_file()
            mod = CustomModule(
                config={**self.test_config, "log_level": mod_lvl}, agent=self.agent
            )
            self.assertEqual(mod.logger.getEffectiveLevel(), mod_lvl_int)
            # todo Fabian discuss, how we can log to a file, or test this better
            # for lvl in LOG_LEVELS:
            #     mod.logger.log(level=logging.getLevelName(lvl), msg="This is a " + lvl)
            #     if mod_lvl_int <= logging.getLevelName(lvl):
            #         if not self._lvl_in_log(lvl=lvl, log=self._read_file()):
            #             raise KeyError(f"{lvl}-log does not work for module-lvl {mod_lvl}")
            #     else:
            #         if self._read_file():
            #             raise KeyError(f"Log should be empty but is not!")
            #     self._clear_file()

    @staticmethod
    def _lvl_in_log(lvl, log):
        lvl = lvl.upper()
        target = f"{lvl} 0s: Agent 'Test', module 'test_mod_id': This is a {lvl}\n"
        return target == log

    def _clear_file(self):
        with open(self.log_filename, "a+") as file:
            file.seek(0)
            file.truncate()

    def _read_file(self):
        with open(self.log_filename, "r") as file:
            return file.read()


if __name__ == "__main__":
    unittest.main()
    logging.basicConfig()
