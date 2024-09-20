"""Module with tests for the limits of the data-broker of the agentlib."""

import time
import unittest

import time

import numpy as np

from agentlib import LocalMASAgency
from agentlib import AgentVariable, BaseModule, BaseModuleConfig


class DataBrokerExplosionTest(BaseModule):
    config: BaseModuleConfig

    def process(self):
        self.agent.data_broker.send_variable(
            AgentVariable(name="exploding_value", value=0)
        )
        yield self.env.event()

    def register_callbacks(self):
        self.agent.data_broker.register_callback(callback=self._resend_variables)

    def _resend_variables(self, variable):
        for i in range(3):
            self.agent.data_broker.send_variable(
                AgentVariable(name="exploding_value", value=i + variable.value)
            )


class Sender(BaseModule):
    config: BaseModuleConfig

    def process(self):
        while True:
            self.agent.data_broker.send_variable(
                AgentVariable(name="some_var", value=self.env.time)
            )
            yield self.env.timeout(0.01)

    def register_callbacks(self):
        pass


class SlowReceiver(BaseModule):
    config: BaseModuleConfig

    def process(self):
        yield self.env.event()

    def register_callbacks(self):
        self.agent.data_broker.register_callback(callback=self._do_some_slow_stuff)

    def _do_some_slow_stuff(self, variable):
        time.sleep(0.5)
        self.logger.info(
            "Time delta of working of variables: %s", self.env.time - variable.value
        )


class FaultyReceiver(BaseModule):
    config: BaseModuleConfig

    def process(self):
        yield self.env.event()

    def register_callbacks(self):
        self.agent.data_broker.register_callback(callback=self._do_some_slow_stuff)

    def _do_some_slow_stuff(self, variable):
        raise Exception


def exploding_modules(until: float, max_queue_size: int):
    agent_configs = [
        {
            "id": "First",
            "max_queue_size": max_queue_size,
            "modules": [
                {
                    "type": {"file": __file__, "class_name": "DataBrokerExplosionTest"},
                    "module_id": "first",
                },
                {
                    "type": {"file": __file__, "class_name": "DataBrokerExplosionTest"},
                    "module_id": "second",
                },
            ],
        }
    ]
    mas = LocalMASAgency(env={"rt": True}, agent_configs=agent_configs)
    mas.run(until=until)


def slow_module(until: float, max_queue_size: int):
    agent_configs = [
        {
            "id": "First",
            "modules": [
                {"type": {"file": __file__, "class_name": "Sender"}},
                {"type": {"file": __file__, "class_name": "SlowReceiver"}},
            ],
            "max_queue_size": max_queue_size,
        }
    ]
    mas = LocalMASAgency(env={"rt": True}, agent_configs=agent_configs)
    mas.run(until=until)


def faulty_module():
    agent_configs = [
        {
            "id": "First",
            "modules": [
                {"type": {"file": __file__, "class_name": "Sender"}},
                {"type": {"file": __file__, "class_name": "FaultyReceiver"}},
            ],
        }
    ]
    mas = LocalMASAgency(env={"rt": True}, agent_configs=agent_configs)
    mas.run(until=10)


class TestDataBrokerLimits(unittest.TestCase):
    def test_slow_module(self):
        slow_module(max_queue_size=10000, until=5)
        with self.assertRaises(RuntimeError):
            slow_module(max_queue_size=100, until=5)

    def test_exploding_module(self):
        exploding_modules(max_queue_size=-1, until=5)
        with self.assertRaises(RuntimeError):
            exploding_modules(max_queue_size=100, until=5)

    def test_faulty_module(self):
        with self.assertRaises(RuntimeError):
            faulty_module()


if __name__ == "__main__":
    unittest.main()
