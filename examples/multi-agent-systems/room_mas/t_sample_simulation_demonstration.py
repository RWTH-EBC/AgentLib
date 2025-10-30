import itertools
import logging
import os
from copy import deepcopy

from agentlib.utils.multi_agent_system import LocalMASAgency

logger = logging.getLogger(__name__)

TRY_CONFIG = {
    "id": "TRYSensorAgent",
    "modules": {
        "sensor": {
            "type": "TRYSensor",
            "t_sample": 60,
            "filename": "data/TRY2015_Aachen_Jahr.dat"
        },
        "ComLocal": {
            "type": "local",
            "parse_json": True
        }
    }
}

ROOM_CONFIG = {
    "id": "Room2Agent",
    "modules": {
        "room": {
            "type": "simulator",
            "model": {
                "type": "fmu",
                "path": "models/SimpleRoom.fmu"
            },
            "t_sample_communication": 50,
            "t_sample_simulation": 15,
            "save_results": True,
            "log_level": "DEBUG",
            "inputs": [
                {
                    "name": "Q_flow_heat",
                    "value": 0
                },
                {
                    "name": "T_oda",
                    "value": 273.15
                }
            ],
            "outputs": [
                {
                    "name": "T_air"
                }
            ]
        },
        "ComLocal": {
            "type": "local",
            "subscriptions": ["BangBangAgent", "TRYSensorAgent"],
            "parse_json": True
        }
    }
}


def run_example(until, with_plots=True, log_level=logging.INFO):
    """
    Runs a multi-agent system (MAS) simulation with heating control agents.

    This function configures and runs a MAS simulation with different control strategies
    (PID and Bang-Bang controllers) for room temperature regulation. It creates multiple
    agents including sensors, room models, and controllers, then simulates their interaction.

    Parameters
    ----------
    until : float
        Simulation end time in seconds.
    with_plots : bool, default=True
        If True, generates and displays plots of simulation results.
    log_level : int, default=logging.INFO
        Logging verbosity level.
    use_direct_callback_databroker : bool, default=False
        Whether to use direct callback execution in the databroker instead of queue-based callbacks.

    """
    # Start by setting the log-level
    logging.basicConfig(level=log_level)

    env_config = {"rt": False, "t_sample": 60, "clock": False}

    # Change the working directly so that relative paths work
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    # create multiple agents with different configurations
    agent_configs = [TRY_CONFIG]

    combinations = [[900, 1800], [60, 900]]

    for t_sample_com, t_sample_sim in itertools.product(*combinations):
        room_cfg = deepcopy(ROOM_CONFIG)
        room_cfg["id"] = f"{t_sample_com}_{t_sample_sim}"
        room_cfg["modules"]["room"]["t_sample_communication"] = t_sample_com
        room_cfg["modules"]["room"]["t_sample_simulation"] = t_sample_sim
        agent_configs.append(room_cfg)

    mas = LocalMASAgency(
        env=env_config,
        agent_configs=agent_configs,
        variable_logging=True,
        use_direct_callback_databroker=False
    )
    # Simulate
    mas.run(until=until)
    # Load results:
    results = mas.get_results(cleanup=True)

    if not with_plots:
        return results

    df_se = results["TRYSensorAgent"]["AgentLogger"]

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2, len(agent_configs) - 1,
        sharex=True, sharey="row", figsize=(16, 8), squeeze=False
    )
    idx = 0
    for t_sample_com, t_sample_sim in itertools.product(*combinations):
        df_ro = results[f"{t_sample_com}_{t_sample_sim}"]["room"]
        # Plot Room agent data for PID controlled zone
        axes[0, idx].plot(df_ro["T_air"], color="red", label="RoomAgent", marker="^")
        axes[1, idx].plot(df_se["T_oda"], color="green", linestyle="-.", label="SensorAgent")
        axes[1, idx].plot(df_ro["T_oda"], color="blue", label="RoomAgent", marker="^")
        # Legend, titles etc:
        axes[0, idx].set_title(
            "$\Delta t_\mathrm{Com}=%s$\n$\Delta t_\mathrm{Sim}=%s$" % (t_sample_com, t_sample_sim)
        )
        axes[-1, idx].set_xlabel("Time / s")
        idx += 1
    axes[0, 0].set_ylabel("$T_{Room}$ / K")
    axes[1, 0].set_ylabel("$T_{oda}$ / K")
    axes[1, 0].legend()
    axes[0, 0].legend()
    fig.tight_layout()

    plt.show()
    return results


if __name__ == "__main__":
    run_example(until=7200, with_plots=True)
