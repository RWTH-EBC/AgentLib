"""
Example of a local multi-agent-system consisting of two agents playing ping
pong. To set up a multi-agent-system by hand, the following steps are required:
    1. Specify the environment configuration
    2. Specify the config of all agents
    3. Create the environment and the agents
    4. Run the simulation by calling the run() method of the environment.
"""

import logging

import pingpong_module
from agentlib.core import Environment, Agent
from agentlib.utils import MultiProcessingBroker

logging.basicConfig(level="DEBUG")
logger = logging.getLogger(__name__)


env_config = {"rt": True, "factor": 1}
agent_config1 = {
    "id": "FirstAgent",
    "modules": [
        {"module_id": "Ag1Com", "type": "multiprocessing_broadcast"},
        {
            "module_id": "Ping",
            "type": {"file": pingpong_module.__file__, "class_name": "PingPong"},
            "start": True,
            "initial_wait": 2,
        },
    ],
}
agent_config2 = {
    "id": "SecondAgent",
    "modules": [
        {"module_id": "Ag2Com", "type": "multiprocessing_broadcast"},
        {
            "module_id": "Pong",
            "type": {"file": pingpong_module.__file__, "class_name": "PingPong"},
        },
    ],
}


if __name__ == "__main__":
    broker = MultiProcessingBroker()
    env = Environment(config=env_config)
    agent1 = Agent(config=agent_config1, env=env)
    agent2 = Agent(config=agent_config2, env=env)
    env.run(until=None)
