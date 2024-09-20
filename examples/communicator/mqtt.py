import sys
import logging
import numpy as np
from agentlib.core import Environment, Agent, datamodels

logging.basicConfig(level="DEBUG")
logger = logging.getLogger(__name__)

env_config = {"rt": True, "factor": 1, "strict": False}
agent_config = {
    "id": "myID",
    "modules": [
        {
            "module_id": "MyComID",
            "type": "mqtt",
            "url": "mqtt://test.mosquitto.org",
            "subtopics": ["/agentlib/#"],
        }
    ],
}

if __name__ == "__main__":
    env = Environment(config=env_config)
    logger.debug(env.config.model_dump_json(indent=2))
    agent = Agent(env=env, config=agent_config)
    logger.debug(agent.config.model_dump_json(indent=2))
    logger.debug(agent.get_module("MyComID").config.model_dump_json(indent=2))

    def callback(message):
        logger.debug(message.json())

    def simulated_output():
        yield env.timeout(delay=0)
        while True:
            output = datamodels.AgentVariable(
                name="MyOutput",
                value=f"current "
                f"envtime : {env.now}"
                f", value: "
                f"{np.random.randint(0, 10)}",
            )
            agent.data_broker.send_variable(output)
            yield env.timeout(delay=1)

    # Create dummy callback for agent
    agent.data_broker.register_callback(callback=callback)

    # Add dummy process to environment
    env.process(simulated_output())
    env.run(until=10)

    # Must be called to close all existing connections
    sys.exit()
