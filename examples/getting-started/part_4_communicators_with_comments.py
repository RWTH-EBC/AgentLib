"""
This is part four of the tutorial.
Here, we will control the model of part three with a PI-controller.
In doing so, we will learn, how to run multiple agents and have them communicate.

In this tutorial we learn about:
 - how to use communicators
 - how to use a PID controller
"""

import logging
from typing import List

import agentlib as ag


# the model is the same as in example three
class HeatedRoomConfig(ag.ModelConfig):
    inputs: List[ag.ModelInput] = [
        ag.ModelInput(name="heating_power_in_watt", value=100)
    ]
    states: List[ag.ModelState] = [
        ag.ModelState(name="temperature_in_celsius", value=20)
    ]
    parameters: List[ag.ModelParameter] = [
        ag.ModelParameter(name="heat_loss_in_watt", value=150),
        ag.ModelParameter(name="thermal_capacity_zone", value=100_000),
    ]
    outputs: List[ag.ModelOutput] = []


class HeatedRoom(ag.Model):
    config: HeatedRoomConfig

    def initialize(self, **kwargs):
        pass

    def do_step(self, *, t_start, t_sample):
        t = self.get("temperature_in_celsius").value  # get/set also works in models
        power = self.get("heating_power_in_watt").value
        loss = self.get("heat_loss_in_watt").value
        capacity = self.get("thermal_capacity_zone").value
        t = t + ((power - loss) / capacity) * t_sample
        self.set("temperature_in_celsius", t)


# here, we have a similar configuration as last time, however, there are differences in
# the variable declarations
process_agent_config = {
    "id": "Process",  # the ID is now Process
    "modules": [
        {
            "module_id": "sim",
            "type": "simulator",
            "model": {
                "type": {"file": __file__, "class_name": "HeatedRoom"},
            },
            "t_sample": 10,
            "save_results": True,
            "result_filename": "results_part3.csv",
            "result_causalities": ["local"],
            "inputs": [
                # notice that we now included an alias in the declaration of the
                # heating power. The alias is the public name of a variable, which is
                # used for communication. The name is important for the
                # internal workings of a module. In this case, the alias has to match
                # with the other agent for communication, while the name has to match
                # with our model implementation.
                # If you do not declare an alias, it will be set to the name. You will
                # see many examples, where this is the case. However it is often useful
                # to be explicit about the alias, to better understand the
                # communication structure of a an agent system
                {"name": "heating_power_in_watt", "value": 200, "alias": "p_heat"}
            ],
            "states": [
                # we also set an alias here. Note that we set the keyword 'shared' to
                # True. The 'shared' keyword is a flag, that is checked in the
                # communicator. A variable will always be exchanged between the modules
                # of an agent, but only sent to other agents, if it is 'shared'.
                {
                    "name": "temperature_in_celsius",
                    "value": 21,
                    "alias": "T_room",
                    "shared": True,
                }
            ],
        },
        # here we define our first communicator, which is of type 'local'. A local
        # communicator can only be used for agents that run on the same Python process.
        # The subscription key contains the list of IDs of the agents, of which we want
        # to receive messages.
        {"module_id": "myComm", "type": "local", "subscriptions": ["PID"]},
    ],
}

pid_agent_config = {
    "id": "PID",  # the ID here matches, what we subscribed to in the process agent
    "modules": [
        {
            "module_id": "myPid",
            "type": "pid",  # pid is the key, to configure the standard PID module
            "setpoint": 21,
            "Kp": 1000,
            "Ti": 10,
            "lb": 0,
            "ub": 500,
            # the PID takes one input and one output, which in control terms are usually
            # u and y. We define the aliases to match the alias in the process agent,
            # while leaving the local names as u and y. Note that we also included the
            # source key here, which is an additional filter. The input will only be
            # updated, if it is sent from an agent with matching source. If the source
            # is not specified, the agent will update the input with any variable that
            # has a matching alias, from any agent. You will not see the source in many
            # examples, however it can be a helpful filter in large mas.
            "input": {"name": "u", "value": 0, "alias": "T_room", "source": "Process"},
            # for the output, we also set an alias. The 'shared'
            # keyword is a bit tricky, because its default value is different depending
            # on the module type and field. For example, here we have to explicitly set
            # the output to 'shared', since a PID module might typically be included in
            # the same agent as the simulation model. For the simulator, the states are
            # not shared by default, but outputs are.
            "output": {
                "name": "y",
                "value": 0,
                "alias": "p_heat",
                "shared": True,
            },
        },
        # here, we use the local communicator as well, matching the ID Process in the
        # subscriptions
        {"module_id": "myComm", "type": "local", "subscriptions": ["Process"]},
    ],
}


def main(with_plots: bool = True):
    logging.basicConfig(level=logging.INFO)
    environment_config = {"rt": False, "factor": 1, "clock": False}
    mas = ag.LocalMASAgency(
        agent_configs=[process_agent_config, pid_agent_config], env=environment_config
    )
    mas.run(1800)

    if with_plots:
        import matplotlib.pyplot as plt
        import pandas as pd

        results = mas.get_results()["my_agent_id"]
        logger_results: pd.DataFrame = results["sim"]
        logger_results["temperature_in_celsius"].plot()
        plt.ylabel("Temperature / °C")
        plt.xlabel("Time / sec")
        plt.show()


if __name__ == "__main__":
    main()
