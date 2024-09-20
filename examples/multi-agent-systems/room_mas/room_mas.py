import logging
import os

from matplotlib.ticker import AutoMinorLocator

from agentlib.utils.multi_agent_system import LocalMASAgency

logger = logging.getLogger(__name__)


def run_example(until, with_plots=True, log_level=logging.INFO):
    # Start by setting the log-level
    logging.basicConfig(level=log_level)

    env_config = {"rt": False, "t_sample": 60, "clock": True}

    # Change the working directly so that relative paths work
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    # create multiple agents with different configuration
    mas = LocalMASAgency(
        env=env_config,
        agent_configs=[
            "configs/TRYSensor.json",
            "configs/Room2.json",
            "configs/Room1.json",
            "configs/PIDAgent.json",
            "configs/BangBangAgent.json",
        ],
        variable_logging=True,
    )
    # Simulate
    mas.run(until=until)
    # Load results:
    results = mas.get_results()

    if not with_plots:
        return results

    df_ro_pid = results["Room1Agent"]["room"]
    df_ro_bb = results["Room2Agent"]["room"]
    df_bb = results["BangBangAgent"]["AgentLogger"]
    df_se = results["TRYSensorAgent"]["AgentLogger"]
    df_pid = results["PIDAgent"]["AgentLogger"]

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, sharex=True)
    # Plot Room agent data for PID controlled zone
    axes[0, 0].plot(df_ro_pid["T_air"], color="blue", label="RoomAgent")
    axes[1, 0].plot(df_ro_pid["Q_flow_heat"], color="blue", label="RoomAgent")
    axes[2, 0].plot(df_ro_pid["T_oda"], color="blue", label="RoomAgent")
    # Plot Room agent data for BangBang controlled zone
    axes[0, 1].plot(df_ro_bb["T_air"], color="blue", label="RoomAgent")
    axes[1, 1].plot(df_ro_bb["Q_flow_heat"], color="blue", label="RoomAgent")
    axes[2, 1].plot(df_ro_bb["T_oda"], color="blue", label="RoomAgent")
    # Plot PID agent data only for model as module has no logging
    axes[0, 0].plot(df_pid["T_air"], color="red", linestyle="--", label="CtrlAgent")
    axes[1, 0].plot(
        df_pid["Q_flow_heat"], color="red", linestyle="--", label="CtrlAgent"
    )
    axes[0, 1].plot(df_bb["T_air"], color="red", linestyle="--", label="CtrlAgent")
    axes[1, 1].plot(
        df_bb["Q_flow_heat"], color="red", linestyle="--", label="CtrlAgent"
    )
    # Plot sensor data for pid case
    axes[2, 0].plot(df_se["T_oda"], color="green", linestyle="-.", label="SensorAgent")
    # Plot sensor data for bangbang case
    axes[2, 1].plot(df_se["T_oda"], color="green", linestyle="-.", label="SensorAgent")
    # Legend, titles etc:
    axes[0, 0].set_ylabel("$T_{Room}$ / K")
    axes[1, 0].set_ylabel("$\dot{Q}_{Room\,,in}$ / W")
    axes[2, 0].set_ylabel("$T_{oda}$ / K")
    axes[2, 0].set_xlabel("Time / s")
    axes[2, 1].set_xlabel("Time / s")
    axes[0, 0].set_title("Zone-1: PID-Controller")
    axes[0, 1].set_title("Zone-2: BangBang-Controller")

    for ax in axes.flatten():
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(
            which="major",
            axis="both",
            linestyle="--",
            linewidth=0.5,
            color="black",
            zorder=0,
        )
        ax.grid(
            which="minor",
            axis="both",
            linestyle="--",
            linewidth=0.5,
            color="0.7",
            zorder=0,
        )

    plt.show()
    return results


if __name__ == "__main__":
    run_example(until=86400 / 10, with_plots=True)
