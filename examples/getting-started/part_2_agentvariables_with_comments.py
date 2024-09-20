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
        # the .name attribute of an AgentVariable is also accessible. Here, we define
        # a local variable for convenience
        name_of_my_number = self.config.my_number.name

        while True:
            # we can access the AgentVariable by its name with the .get() method,
            # defined in the BaseModule and available through inheritance.
            # We accessed the AgentVariable - specifically its name attribute - above
            # through the config. However that is a static copy. With the get() method,
            # we obtain the variable with its current value.
            my_number = self.get(name_of_my_number)

            # we access the variables value, and add our predefined number to it. It
            # is easy to forget you need to access the value, since the get() method
            # returns the whole AgentVariable.
            new_value = my_number.value + self.config.increment

            # Now we log what we did, to confirm it is working.
            self.logger.info(
                f"I will change the value of '{name_of_my_number}' from {my_number.value} to {new_value}."
            )

            # The set method will write the new value of our variable. Using the set
            # method notifies other modules, that we changed the value of this variable.
            # If we were to just assign a value to the variable that we retrieved with
            # get(), it would not be remembered for the next time, and cannot be used
            # by other modules.
            self.set(name_of_my_number, new_value)
            yield self.env.timeout(1)

    def register_callbacks(self):
        pass


agent_config = {
    "id": "my_agent_id",
    "modules": [
        {
            "type": {"file": __file__, "class_name": "MyModule"},
            # when specifying fields of a module config that represent AgentVariables,
            # we have to provide more than just a value. The 'name' key is always
            # required. The 'value' key is not required and will default to None.
            # However, it often results in TypeErrors within the Module, if the value
            # is None, and depends on the specific code where the variable is used.
            "my_number": {"name": "greg", "value": 5},
        },
        # We define a second module in this agent, which will log the value of
        # AgentVariables.
        {
            # The module_id is another key, that is available to all modules. It will
            # default to the class name. Here, we specify it, because we need to know
            # it, when we plot our results below.
            "module_id": "my_logger",
            # Here we use the first method to specify a module type - we simply provide
            # the identifier of a core module. Its source code is located
            # at agentlib/modules/utils/agent_logger.py
            "type": "agentlogger",
            # Here, we specify the time intervall, for which we want to log the value
            # of our AgentVariables.
            "t_sample": 1,
        },
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

    # play around with this example!
    # 1. Change the name of the AgentVariable
    # 2. Change the increment

    # Why do we use an AgentVariable? The Module is a normal Python class after all,
    # can I not just assign a normal instance variable, like so?
    def alternate_process(self):
        self.my_number = self.config.my_number
        while True:
            self.my_number += self.config_increment
            self.env.timeout(1)

    # Answer:
    # You can certainly use normal variables, and class instance variables in modules.
    # In fact, it is often very appropriate to do so.
    #
    # AgentVariables are important, when the variable needs to be shared with other
    # modules or agents.
    # For example, here we use an AgentVariable, so that the
    # agentlogger module can see the value of our variable and keep track of it.
    # We will learn more on how that works later.
