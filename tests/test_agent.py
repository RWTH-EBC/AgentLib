"""Module with tests for the core module of the agentlib."""

import unittest
import json
import os
import threading
import time
import sys

from pydantic import ValidationError

from agentlib.core import (
    Agent,
    AgentVariable,
    AgentVariables,
    Environment,
    DataBroker,
    BaseModule,
    BaseModuleConfig,
)
from agentlib.core.agent import AgentConfig
from agentlib.core.errors import ConfigurationError


class UpdateModuleConfigTest(BaseModuleConfig):
    variables: AgentVariables = []


class UpdateModuleTest(BaseModule):
    config: UpdateModuleConfigTest

    def register_callbacks(self):
        pass

    def process(self):
        yield self.env.event()


class HealthCheckTest(BaseModule):
    config: BaseModuleConfig

    def register_callbacks(self):
        self.agent.data_broker.register_callback(callback=self._cb)

    def process(self):
        while True:
            self.agent.data_broker.send_variable(AgentVariable(name="dummy"))
            yield self.env.timeout(0.1)

    def _cb(self, _):
        if self.env.now > 0.2:
            raise Exception("Module is dead now.")  # raise any Exception


class TestAgent(unittest.TestCase):
    """Class to test only the module settings."""

    def setUp(self) -> None:
        """Setup the test-logger."""
        self.ag_id = "test_ag_id"
        self.mo_id = "test_mo_id"
        self.ag_config = {
            "modules": [],
            "id": self.ag_id,
        }
        self.env_cfg = {"rt": False}
        self.env = Environment(config=self.env_cfg)
        self.filepath = os.path.join(os.getcwd(), "ag_config.json")

    def test_settings_modules_error(self):
        """Test if setting anything but dict or filepath setting works"""
        with self.assertRaises(ValidationError):
            AgentConfig(id=self.ag_id, modules=["Not a filepath"])
        with self.assertRaises(ValidationError):
            AgentConfig(id=self.ag_id, modules=[["Not a dict"]])

    def test_filepath_module_settings(self):
        """Test setting module by filepath"""
        with open(self.filepath, "w+") as file:
            json.dump(self._generate_ag_config()["modules"][0], file)
        AgentConfig(id=self.ag_id, modules=[self.filepath])

    def test_getters(self):
        ag = Agent(config=self.ag_config, env=self.env)
        self.assertEqual(self.ag_id, ag.id)
        self.assertIsInstance(ag.data_broker, DataBroker)
        self.assertIsInstance(ag.config, AgentConfig)
        self.assertIsInstance(str(ag), str)
        self.assertEqual(ag.env, self.env)
        self.assertEqual(ag.modules, [])
        self.assertIsNone(ag.get_module("Not in there"))

    def test_set_config(self):
        """Test the in code setting configs."""
        ag = Agent(config=self.ag_config, env=self.env)
        self.assertEqual(ag.modules, [])
        # Set the config
        ag.config = self._generate_ag_config(module_id=self.mo_id, value=0)
        self.assertEqual(ag.get_module(self.mo_id).variables[0].value, 0)
        # Now with new value
        ag.config = self._generate_ag_config(module_id=self.mo_id, value=10)
        self.assertEqual(ag.get_module(self.mo_id).variables[0].value, 10)

    def test_duplicate_modules(self):
        """Test if duplicate module ids raise KeyError"""
        ag_cfg = self._generate_ag_config(module_id=self.mo_id, value=0)
        ag_cfg["modules"] = ag_cfg.get("modules") + ag_cfg.get("modules")
        with self.assertRaises(KeyError):
            Agent(env=self.env, config=ag_cfg)

    def test_config_setter(self):
        """Test the config setter in agent"""
        ag = Agent(config=self.ag_config, env=self.env)
        ag.config = AgentConfig(**self.ag_config)
        ag.config = ag.config.model_dump_json()
        with open(self.filepath, "w+") as file:
            json.dump(self.ag_config, file)
        ag.config = self.filepath
        # Test if new setup with a file-path works
        Agent(config=self.filepath, env=self.env)
        # Test if new setup with a json-string works
        Agent(config=json.dumps(self.ag_config), env=self.env)
        with self.assertRaises(ConfigurationError):
            Agent(config=json.dumps(self.ag_config) + "SomeFault", env=self.env)

    def test_register_modules(self):
        """Test register bugs in modules"""
        with self.assertRaises(TypeError):
            config = self._generate_ag_config(type_str=("Not", "a dict"))
            Agent(config=config, env=self.env)
        config = self._generate_ag_config(module_id=None)
        env = Environment(config=self.env_cfg)
        ag = Agent(config=config, env=env)
        self.assertEqual(ag.modules[0].id, "UpdateModuleTest")
        with self.assertRaises(ModuleNotFoundError):
            config = self._generate_ag_config(type_str="Not a known module")
            env = Environment(config=self.env_cfg)
            Agent(config=config, env=env)

    def test_health_check(self):
        """Test to see if the health check works."""
        cfg = {
            "modules": [{"type": {"file": __file__, "class_name": "HealthCheckTest"}}],
            "id": self.ag_id,
            "check_alive_interval": 0.01,  # More narrow checking for rt-tests.
        }
        # if rt = False, no thread is used and the Exception is raised.
        # Else, the RunTimeError should be raised by the agent.
        env_cfg = {"rt": True}
        ag = Agent(config=cfg, env=Environment(config=env_cfg))
        ag.env.run(until=0.1)  # 0.1 is before 0.2 and thus raises no error
        # Now catch the error:
        # First try normal operation
        ag = Agent(config=cfg, env=Environment(config=env_cfg))
        with self.assertRaises(RuntimeError):
            # 0.3 is after 0.2 and thus raises the error
            ag.env.run(until=0.5)

        # Try with a daemon thread
        self._stop_test = False
        _thread = threading.Thread(
            target=self._no_daemon_thread, daemon=False, name="notADaemon"
        )
        _thread.start()
        ag = Agent(config=cfg, env=Environment(config={"rt": True}))
        ag.register_thread(thread=_thread)
        try:
            ag.env.run(until=0.31)
        except RuntimeError:
            # The agent should indicate it's dead:
            self.assertFalse(ag.is_alive)
            # The thread should keep running:
            self._still_running = False
            time.sleep(0.1)  # Give the thread some time to set still_running to True
            self.assertTrue(self._still_running)
        finally:
            self._stop_test = True
            _thread.join()

    def _no_daemon_thread(self):
        """
        This dummy thread overwrites a private attribute
        to indicate the thread is still running.
        """
        while True:
            self._still_running = True
            if self._stop_test:
                sys.exit()

    def _generate_ag_config(self, module_id=None, value=0, type_str=None):
        """Helper function to avoid duplicate code"""
        if type_str is None:
            type_str = {"file": __file__, "class_name": "UpdateModuleTest"}
        module = {
            "type": type_str,
            "variables": [{"name": "TestParam", "value": value}],
        }
        if module_id is not None:
            module["module_id"] = module_id
        return {"modules": [module], "id": self.ag_id}

    def tearDown(self) -> None:
        """Delete file"""
        try:
            os.remove(self.filepath)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    unittest.main()
