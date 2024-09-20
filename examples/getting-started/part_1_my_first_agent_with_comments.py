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


# every
class MyFirstModuleConfig(ag.BaseModuleConfig):
    name: str
    age: int = 2


class MyFirstModule(ag.BaseModule):
    config: (
        MyFirstModuleConfig  # this links the functions to the config! easy to forget
    )

    def process(self):
        # the process defines recurring tasks. It usually looks something like this
        while True:
            # we can access the attributes from the config we defined above like this
            self.logger.info(f"I am {self.config.name} with age {self.config.age}.")
            yield self.env.timeout(1)  # timeout is in seconds

    def register_callbacks(self):
        # we will learn about callbacks later
        pass


# the config of an agent can be a python dict or a JSON-file
# the config has two keys: id and modules
agent_config = {
    # the id is the public name of the agent. It will become later, when using
    # communicators. It is also displayed in log messages.
    "id": "my_agent_id",
    # second comes a list of modules, which define the function of the agent. Usually,
    # an agent has a communicator and a functional module, but in fact, any
    # combination of modules is possible.
    # In this example, there is no communication with other agents, so we only have one
    # module
    "modules": [
        {
            # the type key defines the class of a module and is always required
            # There are two ways to specify the type
            #   1. Use a module from the core, or a plugin. You only need to write down
            #   the string-key which maps to it.
            #   2. Use a custom module. Under type, write down a dict, pointing to the
            #   exact python class you want to execute.
            # In our example, we use option two, to load the module defined above.
            "type": {"file": __file__, "class_name": "MyFirstModule"},
            # on the same level as type, the remaining configuration of the module has
            # to be done. For our module defined above, we defined a 'name' field which
            # we now have to specify
            "name": "first_agent",
        }
    ],
}


def main():
    # We have to make sure, our log level is set to INFO, otherwise, our process
    # defined above will not print anything
    logging.basicConfig(level=logging.INFO)

    # every agent runs in an environment. The environment has four keys:
    # 1. 'rt': If True, the code will run in realtime
    # 2. 'factor': Speed up realtime if <1, slow down if >1. Ignored if 'rt' is False
    # 3. 'clock': If True, periodically print the current simulation time
    # 4. 't_sample': Intervall for the clock. 3 and 4 only affect logs, not results
    environment_config = {"rt": False, "factor": 1, "clock": False, "t_sample": 60}
    env = ag.Environment(config=environment_config)

    # create the agent. Multiple agents can be created with the same environment.
    agent = ag.Agent(config=agent_config, env=env)

    # runs the environment for 10 seconds, starting the agent
    env.run(10)


if __name__ == "__main__":
    main()

    # play around with this example!
    # 1. Try changing the name of the agent.
    # 2. Try changing the age of the agent.
    #
    # Bonus: Add a field to the agent config which lets you change the timeout through
    # the configuration dictionary
