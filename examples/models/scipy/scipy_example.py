import logging
from agentlib.models.scipy_model import ScipyStateSpaceModel

#  t: start time, inputs: system inputs u, states: state variables x, outputs: system outputs y on highest level",


logger = logging.getLogger(__name__)
UA = 100
C_p = 1000

config_state_space = {
    "description": "This is an example description for an scipy-state-space-model",
    "dt": 0.1,
    "inputs": [
        {
            "name": "T_oda",
            "value": 273.15,
            "type": "float",
            "unit": "K",
            "description": "Outdoor air temperature",
        },
        {
            "name": "Q_flow_heat",
            "value": 0,
            "type": "float",
            "unit": "W",
            "description": "Internal Gains",
        },
    ],
    "states": [
        {"name": "T_room", "value": 293.15, "description": "Room temperature"},
    ],
    "outputs": [{"name": "T_room", "description": "Room temperature"}],
    "system": {
        "A": [-UA / C_p],
        "B": [UA / C_p, 1 / C_p],
        "C": [1],
        "D": [0, 0],
    },
}


def run_example(with_plots: bool = True):
    model = ScipyStateSpaceModel(**config_state_space)
    logger.info(model.config.system)

    y = []
    for i in range(100):
        model.do_step(t_start=i, t_sample=1)
        y.append(model.outputs[0].value)

    if with_plots:
        import matplotlib.pyplot as plt

        plt.plot(y)
        plt.show()


if __name__ == "__main__":
    run_example(with_plots=True)
