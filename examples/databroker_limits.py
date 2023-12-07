import logging
import time

from agentlib import LocalMASAgency
from agentlib import AgentVariable, BaseModule, BaseModuleConfig


class DataBrokerExplosionTest(BaseModule):
    config: BaseModuleConfig

    def process(self):
        self.agent.data_broker.send_variable(AgentVariable(name="exploding_value", value=0))
        yield self.env.event()

    def register_callbacks(self):
        self.agent.data_broker.register_callback(callback=self._resend_variables)

    def _resend_variables(self, variable):
        for i in range(3):
            self.agent.data_broker.send_variable(AgentVariable(name="exploding_value", value=i + variable.value))


class Sender(BaseModule):
    config: BaseModuleConfig

    def process(self):
        while True:
            self.agent.data_broker.send_variable(AgentVariable(name="some_var", value=self.env.time))
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
        self.logger.info("Time delta of working of variables: %s", self.env.time - variable.value)


class FaultyReceiver(BaseModule):
    config: BaseModuleConfig

    def process(self):
        yield self.env.event()

    def register_callbacks(self):
        self.agent.data_broker.register_callback(callback=self._do_some_slow_stuff)

    def _do_some_slow_stuff(self, variable):
        raise Exception


def exploding_modules():
    agent_configs = [
        {
            "id": "First",
            "modules": [
                {"type": {"file": __file__, "class_name": "DataBrokerExplosionTest"}, "module_id": "first"},
                {"type": {"file": __file__, "class_name": "DataBrokerExplosionTest"}, "module_id": "second"}
            ]
        }
    ]
    MAS = LocalMASAgency(env={"rt": True}, agent_configs=agent_configs)
    MAS.run(until=3)


def slow_module():
    agent_configs = [
        {
            "id": "First",
            "modules": [
                {"type": {"file": __file__, "class_name": "Sender"}},
                {"type": {"file": __file__, "class_name": "SlowReceiver"}}
            ]
        }
    ]
    MAS = LocalMASAgency(env={"rt": True}, agent_configs=agent_configs)
    MAS.run(until=5)


def faulty_module():
    agent_configs = [
        {
            "id": "First",
            "modules": [
                {"type": {"file": __file__, "class_name": "Sender"}},
                {"type": {"file": __file__, "class_name": "FaultyReceiver"}}
            ]
        }
    ]
    MAS = LocalMASAgency(env={"rt": True}, agent_configs=agent_configs)
    MAS.run(until=1)


if __name__ == '__main__':
    logging.basicConfig(level="DEBUG")
    # slow_module()
    exploding_modules()
    # faulty_module()