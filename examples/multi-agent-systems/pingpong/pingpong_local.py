import logging
import pingpong_module
from agentlib.utils.multi_agent_system import LocalMASAgency

logging.basicConfig(level="DEBUG")
logger = logging.getLogger(__name__)

env_config = {"rt": True, "factor": 0.6, "strict": True}

agent_config1 = {
    "id": "FirstAgent",
    "modules": [
        {"module_id": "Ag1Com", "type": "local", "subscriptions": ["SecondAgent"]},
        {
            "module_id": "Ping",
            "type": {"file": pingpong_module.__file__, "class_name": "PingPong"},
            "start": True,
        },
    ],
}

agent_config2 = {
    "id": "SecondAgent",
    "modules": [
        {"module_id": "Ag2Com", "type": "local", "subscriptions": ["FirstAgent"]},
        {
            "module_id": "Pong",
            "type": {"file": pingpong_module.__file__, "class_name": "PingPong"},
        },
    ],
}


if __name__ == "__main__":
    # Add configs to LocalMAS Agency
    MAS = LocalMASAgency(agent_configs=[agent_config1, agent_config2], env=env_config)
    MAS.run(until=None)
