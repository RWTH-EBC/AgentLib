"""Module with tests for the core module of the agentlib."""

import json
import os
import unittest
import uuid
import sys

from agentlib import BaseModuleConfig, ModelConfig
from agentlib.core.agent import AgentConfig
from agentlib.core.errors import ConfigurationError
from agentlib.utils import validators, custom_injection
from agentlib.utils.broker import Broker
from agentlib.utils.load_config import load_config


class Foo:
    """DummyImport class"""

    test_value = 5


class TestUtil(unittest.TestCase):
    """Class to test only the module settings."""

    def test_conv_list(self):
        """Test the convert list function"""
        self.assertIsInstance(validators.convert_to_list(None), list)
        self.assertIsInstance(validators.convert_to_list(1), list)
        self.assertIsInstance(validators.convert_to_list([1, 2]), list)

    def test_broker(self):
        """Test the broker"""
        broker = Broker()
        broker.register_client(10)
        broker.delete_client(10)
        broker.delete_client("Not the 10")

    def test_custom_injection(self):
        """Test the custom injection function"""
        config = {"file": __file__, "class_name": "Foo"}
        module = custom_injection(config=config)
        self.assertEqual(module().test_value, 5)
        # Test module_name.
        uuid_4 = str(uuid.uuid4())
        custom_injection(config=config, module_name=uuid_4)
        self.assertTrue(uuid_4 in sys.modules)
        # Test existing modules. E.g. 'sys' should not be overwritten
        with self.assertRaises(ImportError):
            custom_injection(config=config, module_name="sys")
        # If class is found in existing module, this class should be returned
        sys_path = custom_injection(
            config={"file": __file__, "class_name": "path"}, module_name="sys"
        )
        self.assertEqual(sys_path, sys.path)


class TestLoadConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.configs = {
            AgentConfig: {"id": "agentid", "modules": []},
            BaseModuleConfig: {
                "type": "dummy",
                "module_id": "myid",
                "_agent_id": "myagent",
            },
            ModelConfig: {},
        }
        self.filepath = "load_config_test.json"

    def tearDown(self) -> None:
        try:
            os.remove(self.filepath)
        except OSError:
            pass

    def test_load_config(self):
        """Tests the load_config utility"""
        for config_type, config in self.configs.items():
            config_json_s = json.dumps(config)
            with open(self.filepath, "w") as f:
                json.dump(config, f)
            config_object = config_type.model_validate(config)
            load_config(config, config_type)
            load_config(config_json_s, config_type)
            load_config(config_object, config_type)
            load_config(self.filepath, config_type)

    def test_load_config_error(self):
        wrong_string = "not_a_path.josn"
        with self.assertRaises(ConfigurationError):
            load_config(wrong_string, config_type=BaseModuleConfig)


if __name__ == "__main__":
    unittest.main()
