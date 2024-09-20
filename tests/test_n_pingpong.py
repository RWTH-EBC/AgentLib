"""
This test file checks if MAS correctly execute.
"""

import itertools
import unittest
import time
from pydantic import Field
import numpy as np
from agentlib.core import BaseModule, Agent, BaseModuleConfig
from agentlib.core.datamodels import AgentVariable
from agentlib.utils import MultiProcessingBroker, LocalBroadcastBroker, LocalBroker
from agentlib.utils.multi_agent_system import LocalMASAgency

# pylint: disable=missing-module-docstring,missing-class-docstring
PORT = 39920


class NPingPongConfig(BaseModuleConfig):
    n: int = Field(description="Trailing number of pingpong agent")
    service: bool = Field(
        default=False, description="Indicates if agent has the first service"
    )
    restart: bool = Field(
        default=False, description="Indicates if agent will restart the cycle"
    )
    initial_wait: float = Field(
        default=0, description="Wait the given amount of seconds before starting."
    )


class NPingPong(BaseModule):
    """Dummy Module to simulate N-Agents playing PingPong"""

    config: NPingPongConfig

    def __init__(self, config: dict, agent: Agent):
        super().__init__(config=config, agent=agent)
        self.finished = False

    def process(self):
        if self.config.service:
            self.logger.debug("Waiting %s s before starting", self.config.initial_wait)
            time.sleep(self.config.initial_wait)
            self.logger.info("%s starts service and sends: Ping_1", self.agent.id)
            self.agent.data_broker.send_variable(
                AgentVariable(
                    name=self.id, value="Ping_1", source=self.source, shared=True
                )
            )
        if self.config.restart:
            while not self.finished:
                pass
            # Once finished, go on and terminate the process:
            yield self.env.timeout(1)
        else:
            yield self.env.event()

    @property
    def n(self):
        return self.config.n

    def register_callbacks(self):
        if self.id == f"Pong_{self.n}":
            alias = f"Ping_{self.n}"
        elif self.id == f"Ping_{self.n}":
            alias = f"Pong_{self.n-1}"
        self.agent.data_broker.register_callback(
            alias=alias, source=None, callback=self._callback
        )

    def _callback(self, variable: AgentVariable):
        self.logger.info("Received: %s", variable.value)
        if self.config.restart:
            answer = "Pong_0"
            self.logger.info("RESTART")
            self.finished = True
        else:
            answer = self.id

        if answer:
            time.sleep(0.00000001)
            self.logger.info("Sends: %s", answer)
            self.agent.data_broker.send_variable(
                AgentVariable(
                    name=self.id, value=answer, source=self.source, shared=True
                )
            )


class TestNPingPong(unittest.TestCase):
    def setUp(self) -> None:
        self.n = np.random.randint(2, 10)

    @unittest.skip
    def test_local_communicators(self):
        """Tests the local communicators in all possible combinations."""
        # todo for some reason this test does not work this way, I get stuck in an infinite loop but I dont know where
        comm_types = ["local", "local_broadcast"]
        parse_json = [True, False]
        rt = [True, False]
        combinations = itertools.product(comm_types, parse_json, rt)
        for com_type, parse_json, rt in combinations:
            broker = LocalBroker()
            broker_bc = LocalBroadcastBroker()
            with self._create_2n_pingpong_agents(
                com_type=com_type, parse_json=parse_json, rt=rt
            ) as mas:
                print("starting")
                mas.run(until=1)
            print(f"completed {com_type, parse_json, rt}")
            broker.delete_all_clients()
            broker_bc.delete_all_clients()

    def test_multiprocessing_broadcast(self):
        """Test the NPingPong system using multiprocessing broadcast as communicator"""
        broker = MultiProcessingBroker(config={"port": PORT})
        with self._create_2n_pingpong_agents(
            com_type="multiprocessing_broadcast"
        ) as mas:
            mas.run(until=1)

    @unittest.skip("MQTT refuses connection in ci")
    def test_mqtt(self):
        """Test the NPingPong system using local_broadcast as communicator"""
        mas = self._create_2n_pingpong_agents(com_type="mqtt")
        mas.run(until=1)

    def _create_2n_pingpong_agents(
        self, com_type="local_broadcast", parse_json=True, rt=True
    ):
        """Create the configs and threads"""
        _pingpong_module = {"file": __file__, "class_name": "NPingPong"}
        configs = []
        for _n in range(1, self.n + 1):
            _com_ping = {
                "module_id": f"AgPingCom_{_n}",
                "type": com_type,
            }
            _com_pong = {
                "module_id": f"AgPongCom_{_n}",
                "type": com_type,
            }

            if com_type not in ["local_broadcast", "multiprocessing_broadcast"]:
                _com_ping["subscriptions"] = [f"AgPong_{_n - 1 if _n > 1 else self.n}"]
                _com_pong["subscriptions"] = [f"AgPing_{_n}"]

            pong_module = {"module_id": f"Pong_{_n}", "type": _pingpong_module, "n": _n}
            ping_module = {"module_id": f"Ping_{_n}", "type": _pingpong_module, "n": _n}
            if com_type == "mqtt":
                _com_ping.update({"url": "mqtt://test.mosquitto.org"})
                _com_pong.update({"url": "mqtt://test.mosquitto.org"})
            if com_type == "multiprocessing_broadcast":
                _com_ping.update({"port": PORT})
                _com_pong.update({"port": PORT})
                ping_module["initial_wait"] = 3
            elif com_type in ["local_broadcast", "local"]:
                _com_ping.update({"parse_json": parse_json})
                _com_pong.update({"parse_json": parse_json})
            elif com_type in ["multiprocessing_broadcast"]:
                pass  # No subs needed
            else:
                raise TypeError(f"Com_type '{com_type}' not supported")

            if _n == 1:
                ping_module.update({"service": True})
            if _n == self.n:
                pong_module.update({"restart": True})
            ping_ag_conf = {"id": f"AgPing_{_n}", "modules": [_com_ping, ping_module]}
            pong_ag_conf = {"id": f"AgPong_{_n}", "modules": [_com_pong, pong_module]}

            configs.append(ping_ag_conf)
            configs.append(pong_ag_conf)

        env_config = {"rt": rt, "factor": 1}
        return LocalMASAgency(
            agent_configs=configs, env=env_config, use_threading=False
        )


if __name__ == "__main__":
    unittest.main()
