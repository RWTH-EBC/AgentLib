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


class HeatedRoom(ag.Model):
    config: HeatedRoomConfig

    def initialize(self, **kwargs):
        pass

    def do_step(self, *, t_start, t_sample):
        t = self.get("temperature_in_celsius").value
        power = self.get("heating_power_in_watt").value
        loss = self.get("heat_loss_in_watt").value
        capacity = self.get("thermal_capacity_zone").value
        t = t + ((power - loss) / capacity) * t_sample
        self.set("temperature_in_celsius", t)


pid_agent_config = {
    "id": "PID",
    "modules": [
        {
            "module_id": "myBangbang",
            "type": "bangbang",
            "gain": 600,
            "lb": 20,
            "ub": 22,
            "input": {"name": "u", "value": 0, "alias": "room_temp"},
            "output": {
                "name": "y",
                "value": 0,
                "alias": "heating_power",
                "shared": True,
            },
        },
        {"module_id": "myLogger", "type": "AgentLogger"},
        {"module_id": "myComm", "type": "local", "subscriptions": ["Process"]},
    ],
}

process_agent_config = {
    "id": "Process",
    "modules": [
        {
            "module_id": "sim",
            "type": "simulator",
            "model": {"type": {"file": __file__, "class_name": "HeatedRoom"}},
            "t_sample": 10,
            "inputs": [
                {"name": "heating_power_in_watt", "value": 0, "alias": "heating_power"}
            ],
            "states": [
                {
                    "name": "temperature_in_celsius",
                    "value": 21,
                    "shared": True,
                    "alias": "room_temp",
                }
            ],
        },
        {"module_id": "myLogger", "type": "AgentLogger"},
        {"module_id": "myComm", "type": "local", "subscriptions": ["PID"]},
    ],
}


def run_example(with_plots=True, log_level=logging.INFO):
    # Set the log-level
    logging.basicConfig(level=log_level)

    env_config = {"rt": False, "factor": 0.01, "clock": True, "t_sample": 60}
    env = ag.Environment(config=env_config)
    pid_agent = ag.Agent(config=pid_agent_config, env=env)
    process_agent = ag.Agent(config=process_agent_config, env=env)
    env.run(until=1000)

    results = process_agent.get_results()
    res = results["myLogger"]
    if with_plots:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 1)
        res["room_temp"].plot(ax=ax[0], label="$T_{room}$")
        ax[0].axhline(20, label="_", linestyle="--", color="black")
        ax[0].axhline(22, label="_", linestyle="--", color="black")

        res["heating_power"].plot(ax=ax[1], label="$\dot{Q}_{heat}$")
        ax[0].legend()
        ax[1].legend()
        plt.show()
    return {"PID_1": results}


if __name__ == "__main__":
    run_example(log_level="INFO")
