"""
This is part three of the tutorial.
Here, we will simulate a simple model.

In this tutorial we learn about:
 - how to use the simulator
 - how to write a simple model
 - how to use the local MAS utility to start your agents


The model will be about the change of a room temperature under a heat load.
The differential equation for the room temperature reads:
dT / dt = ((power - loss) / capacity)
The integration of this differential equation has to be defined in the model itself.
Here, we use a simple Euler-step for integration.

While we use a custom-written model in this example, there are other model types which
can be very useful and come with pre-implemented numeric routines. For example, the
FMU-model can simulate an FMU, and in the agentlib_mpc plugin, there is a CasADi-model,
 which handles the integration of ODEs for you.

We use the predefined simulator module. This can be a bit intimidating, as the
simulator has many configuration parameters, of which we may not know what they do
and which ones we will need. The source code of the simulator is found
at agentlib/modules/simulator.py.
"""

import logging
from typing import List

import agentlib as ag


# define the inputs, outputs, states and parameters of your model
from agentlib.utils.multi_agent_system import LocalMASAgency


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
    config: HeatedRoomConfig  # match the config to the model

    def initialize(self, **kwargs):
        pass

    # define a discrete simulation step
    def do_step(self, *, t_start, t_sample):
        t = self.get("temperature_in_celsius").value  # get/set also works in models
        power = self.get("heating_power_in_watt").value
        loss = self.get("heat_loss_in_watt").value
        capacity = self.get("thermal_capacity_zone").value
        t = t + ((power - loss) / capacity) * t_sample
        self.set("temperature_in_celsius", t)


agent_config = {
    "id": "my_agent_id",
    "modules": [
        {
            "module_id": "sim",
            "type": "simulator",  # specify the simulator module
            "model": {
                # under type, we include a custom injection to our defined model
                # other options for type are:
                # "fmu" (see examples/multi-agent-systems/room_mas/configs/Room1.json)
                # or "statespace"
                "type": {"file": __file__, "class_name": "HeatedRoom"},
            },
            # specify the interval, in which outputs are written, and inputs are updated
            "t_sample": 10,
            # the simulator can save its own results if desired, there is no need for an AgentLogger
            "save_results": True,
            # results are stored in RAM if not specified
            "result_filename": "results_part3.csv",
            # define, which variable groups we want to store in the results. Note that
            # here, we wrote locals, which maps to the states. This is a special case,
            # where we used states instead of locals in some places, to avoid shadowing
            # the Python builtin locals. By default, we store inputs and outputs. The
            # locals (i.e. states) are not stored by default, as they are often of
            # extremely high number in FMU models. Therefore, we have to activate them
            # manually here.
            "result_causalities": ["local"],
            # the inputs and states fields of the simulator are lists of AgentVariables
            # There are also the fields outputs and parameters. We do not have to
            # specify all the variables that are given in the model config. However,
            # only variables that we specify here, can be have their values changed
            # from the default in the model, and only outputs defined here can be
            # sent to other agents.
            "inputs": [
                {"name": "heating_power_in_watt", "value": 200},
            ],
            "states": [
                {"name": "temperature_in_celsius", "value": 21},
            ],
        },
    ],
}


def main(with_plots: bool = True):
    logging.basicConfig(level=logging.INFO)

    environment_config = {"rt": False, "factor": 1, "clock": False}
    # we use the LocalMASAgency class to start our agent
    # It can be passed a list of agent configs, and an environment config and will run
    # all agents in this environment
    mas = ag.LocalMASAgency(agent_configs=[agent_config], env=environment_config)

    # the run command now is part of the mas
    mas.run(1800)

    if with_plots:
        import matplotlib.pyplot as plt
        import pandas as pd

        # there is also a results function for the MAS, we can get agents with their id
        results = mas.get_results()["my_agent_id"]

        # key here is the module_id of the simulator
        logger_results: pd.DataFrame = results["sim"]

        # key is name of the variable (it's actually the alias, more on this later!)
        logger_results["temperature_in_celsius"].plot()
        plt.ylabel("Temperature / °C")
        plt.xlabel("Time / sec")
        plt.show()


if __name__ == "__main__":
    main()

    # play around with this example!
    # 1. Try changing the values of the parameters
    #     a) through the ModelConfig class
    #     b) in the agent config dict
    # 2. Try running the example in Realtime. Consider using the factor of the
    #    environment, to run in scaled real time
    # Bonus: Try adding an output to the model. The output could be something simple, like the room temperature, only
    # this time in kelvin instead of celsius.How would you do that? Can you plot it
    # as well?
