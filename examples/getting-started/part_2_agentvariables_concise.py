"""
This is part two of the tutorial.
Here, we will use AgentVariables to specify properties of the agent.

In this tutorial we learn about:
 - How to introduce state into an agent with AgentVariables
 - How to log AgentVariables and plot them over time

"""

import logging

import agentlib as ag


class MyModuleConfig(ag.BaseModuleConfig):
    increment: float = 0.5
    my_number: ag.AgentVariable


class MyModule(ag.BaseModule):
    config: MyModuleConfig

    def process(self):
        name_of_my_number = self.config.my_number.name
        while True:
            my_number = self.get(name_of_my_number)
            new_value = my_number.value + self.config.increment
            self.logger.info(
                f"I will change the value of '{name_of_my_number}' from {my_number.value} to {new_value}."
            )
            self.set(name_of_my_number, new_value)
            yield self.env.timeout(1)

    def register_callbacks(self):
        pass


agent_config = {
    "id": "my_agent_id",
    "modules": [
        {
            "type": {"file": __file__, "class_name": "MyModule"},
            "my_number": {"name": "greg", "value": 5},
        },
        {"module_id": "my_logger", "type": "agentlogger", "t_sample": 1},
    ],
}


def main(with_plots: bool = True):
    logging.basicConfig(level=logging.INFO)

    environment_config = {"rt": False, "factor": 1, "clock": False}
    env = ag.Environment(config=environment_config)
    agent = ag.Agent(config=agent_config, env=env)
    env.run(10)

    if with_plots:
        import matplotlib.pyplot as plt
        import pandas as pd

        results = agent.get_results()

        # key here is the module_id of the agentlogger
        logger_results: pd.DataFrame = results["my_logger"]

        # key is name of the variable (it's actually the alias, more on this later!)
        logger_results["greg"].plot()
        plt.show()


if __name__ == "__main__":
    main()
