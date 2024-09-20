"""Module to test the Environment in the agentlib"""

import unittest
import time
import os
import json
from pydantic import ValidationError
from agentlib.core import Environment
from agentlib.core.environment import EnvironmentConfig, make_env_config


class TestEnvironment(unittest.TestCase):
    """Class with tests for Environment"""

    def setUp(self) -> None:
        self.config = {
            "rt": False,
            "factor": 1.0,
            "strict": True,
            "t_sample": 1,
            "offset": 1,
        }
        self.filepath = os.path.join(os.getcwd(), "config.json")

    def test_initial_time(self):
        """Test if negative initial_time raises error"""
        config = self.config.copy()
        config["initial_time"] = -1
        with self.assertRaises(ValidationError):
            Environment(config=config)

    def test_normal_setup(self):
        """Test normal setup of environment"""
        Environment(config=self.config)

    def test_config_setter(self):
        """Test the setter of config"""
        env = Environment(config=self.config)
        env._config = make_env_config(self.config)
        self.assertIsInstance(env.config, EnvironmentConfig)
        # Test parse file
        with open(self.filepath, "w+") as file:
            json.dump(self.config, file)
        env._config = make_env_config(self.filepath)
        self.assertIsInstance(env.config, EnvironmentConfig)

    def test_rt(self):
        """Test rt property"""
        config = self.config.copy()
        config["rt"] = True
        env = Environment(config=config)
        env.run(until=2)
        self.assertGreaterEqual(env.time, time.time())

    def test_no_config(self):
        """Test no config setup"""
        env = Environment()
        self.assertIsInstance(env.config, EnvironmentConfig)

    def tearDown(self) -> None:
        """Delete created files"""
        try:
            os.remove(self.filepath)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    unittest.main()
