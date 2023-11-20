"""
This is part three of the tutorial.
Here, we will simulate a simple model.

In this tutorial we learn about:
 - how to use the simulator
 - how to write a simple model
"""

import logging
from typing import List

import agentlib as ag


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


agent_config = {
    "id": "my_agent_id",
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
            "inputs": [{"name": "heating_power_in_watt", "value": 200}],
            "states": [{"name": "temperature_in_celsius", "value": 21}],
        },
    ],
}


def main(with_plots: bool = True):
    logging.basicConfig(level=logging.INFO)
    environment_config = {"rt": False, "factor": 1, "clock": False}
    mas = ag.LocalMASAgency(agent_configs=[agent_config], env=environment_config)
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
