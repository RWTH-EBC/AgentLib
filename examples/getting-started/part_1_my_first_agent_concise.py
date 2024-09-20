"""
This is part one of the tutorial.
Basic example to execute a simple agent, which logs its name to the console.

In this tutorial we learn about:
 - creating our first agent and running it for 10 seconds
 - creating a module which logs some text
 - processes
 - agent configuration
"""

import logging

import agentlib as ag


class MyFirstModuleConfig(ag.BaseModuleConfig):
    name: str
    age: int = 2


class MyFirstModule(ag.BaseModule):
    config: MyFirstModuleConfig

    def process(self):
        while True:
            self.logger.info(f"I am {self.config.name} with age {self.config.age}.")
            yield self.env.timeout(1)

    def register_callbacks(self):
        pass


agent_config = {
    "id": "my_agent_id",
    "modules": [
        {
            "type": {"file": __file__, "class_name": "MyFirstModule"},
            "name": "first_agent",
        }
    ],
}


def main():
    logging.basicConfig(level=logging.INFO)
    environment_config = {"rt": False, "factor": 1, "clock": False, "t_sample": 60}
    env = ag.Environment(config=environment_config)
    agent = ag.Agent(config=agent_config, env=env)
    env.run(10)


if __name__ == "__main__":
    main()
