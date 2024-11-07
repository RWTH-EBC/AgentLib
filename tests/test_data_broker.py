"""Module with tests for the core module of the agentlib."""

import time
import unittest

import numpy as np

from agentlib import BaseModule, Agent, BaseModuleConfig
from agentlib.core import (
    LocalDataBroker,
    AgentVariable,
    Source,
    Environment,
    RTDataBroker,
)
from agentlib.core.logging_ import CustomLogger


class TestLocalDataBroker(unittest.TestCase):
    """Class to test only the module settings."""

    def setUp(self) -> None:
        """Setup the test-logger."""
        env = Environment()
        self.data_broker = LocalDataBroker(
            logger=CustomLogger("TestLocalDataBroker", env=env),
            env=env,
        )
        self.n_vars = np.random.randint(1, 20)
        self.counter = 0

    def perform_callbacks(self):
        for _ in range(self.data_broker._variable_queue.qsize()):
            self.data_broker._execute_callbacks()

    def test_callback(self):
        """Test (de)-registering callbacks"""
        self.data_broker.register_callback(
            alias="Test", source=None, callback=self.dummy_callback
        )
        self.assertEqual(len(self.data_broker._unmapped_callbacks), 1)
        # Send not matching inputs
        for _ in range(self.n_vars):
            self.data_broker.send_variable(variable=AgentVariable(name="1"))
        self.assertEqual(self.counter, 0)
        # Send matching inputs
        for _ in range(self.n_vars):
            self.data_broker.send_variable(variable=AgentVariable(name="Test"))
        self.perform_callbacks()
        self.assertEqual(self.counter, self.n_vars)
        self.data_broker.deregister_callback(
            alias="Test", source=None, callback=self.dummy_callback
        )
        self.assertEqual(len(self.data_broker._unmapped_callbacks), 0)
        # Register matching input
        source = Source(agent_id="1", module_id="2")
        self.data_broker.register_callback(
            alias="Test", source=source, callback=self.dummy_callback
        )
        self.assertEqual(len(self.data_broker._mapped_callbacks), 1)
        self.counter = 0
        for _ in range(self.n_vars):
            self.data_broker.send_variable(variable=AgentVariable(name="Test"))
        self.assertEqual(self.counter, 0)
        for _ in range(self.n_vars):
            self.data_broker.send_variable(
                variable=AgentVariable(name="Test", source=source)
            )
        self.perform_callbacks()
        self.assertEqual(self.counter, self.n_vars)
        self.data_broker.deregister_callback(
            alias="Test", source=source, callback=self.dummy_callback
        )
        key, val = self.data_broker._mapped_callbacks.popitem()
        self.assertEqual(val, [])

    def dummy_callback(self, _):
        self.counter += 1

    def test_trigger_recursion_error(self):
        self.never_reached = True
        self.data_broker.register_callback(
            alias="Test_2", source=None, callback=self.recursion_callback_1
        )
        self.data_broker.register_callback(
            alias="Test_1", source=None, callback=self.recursion_callback_2
        )
        self.data_broker.send_variable(AgentVariable(name="Test_2"))
        while self.never_reached:
            self.perform_callbacks()
        self.assertFalse(self.never_reached)

    def recursion_callback_1(self, variable):
        self.data_broker.send_variable(AgentVariable(name="Test_1"))

    def recursion_callback_2(self, variable):
        self.data_broker.send_variable(AgentVariable(name="Test_2"))
        self.never_reached = False

    def test_modules_cannot_overwrite_each_others_callbacks(self):
        """Makes sure that all checks are done, to prevent modules from deregistering
        other modules callbacks accidentally."""
        alias = "test_alias"
        name = "test_name"
        test_config = {
            "type": "not_necessary_here",
        }
        agent = Agent(config={"id": "Test", "modules": []}, env=Environment())

        class Module(BaseModule):
            config: BaseModuleConfig

            def register_callbacks(self): ...
            def process(self):
                yield self.env.event()

        module_1 = Module(agent=agent, config={**test_config, "module_id": "mod1"})
        module_2 = Module(agent=agent, config={**test_config, "module_id": "mod2"})

        self.data_broker.register_callback(
            alias=alias, source=None, callback=module_1._callback_config_vars, name=name
        )

        # module 2 tries to deregister the same callback, but it should not work since it is a different module
        self.data_broker.deregister_callback(
            alias=alias, source=None, callback=module_2._callback_config_vars, name=name
        )
        self.assertEqual(len(self.data_broker._unmapped_callbacks), 1)
        # module 1 deregisters the callback, which should work
        self.data_broker.deregister_callback(
            alias=alias, source=None, callback=module_1._callback_config_vars, name=name
        )
        self.assertEqual(len(self.data_broker._unmapped_callbacks), 0)

    def test_process(self):
        """Tests that the receive method is properly called during a process."""
        env = self.data_broker.env
        timeout = 1
        until = 10

        self.data_broker.register_callback(
            alias="Test_1",
            source=None,
            callback=self.dummy_callback,
        )

        def my_process():
            while True:
                self.data_broker.send_variable(AgentVariable(name="Test_1"))
                yield env.timeout(timeout)

        env.process(my_process())
        env.run(until=until)

        self.assertEqual(int(until / timeout), self.counter)


class TestRTDataBroker(TestLocalDataBroker):
    def setUp(self) -> None:
        """Setup the test-logger."""
        env = Environment(config={"rt": True})
        self.data_broker = RTDataBroker(
            logger=CustomLogger("TestLocalDataBroker", env=env), env=env
        )
        next(self.data_broker._start_executing_callbacks(env))
        self.n_vars = np.random.randint(1, 20)
        self.counter = 0

    def perform_callbacks(self):
        time.sleep(0.1)

    def test_process(self):
        # skip the process test for RealTime
        ...


if __name__ == "__main__":
    unittest.main()
