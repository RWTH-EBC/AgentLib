import logging
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import agentlib as ag


def create_sample_data():
    date_today = datetime.now()
    time = pd.date_range(date_today, date_today + timedelta(minutes=10), freq="30S")
    data1 = np.sin(np.linspace(0, 2 * np.pi, len(time))) * 10 + 20
    data2 = np.cos(np.linspace(0, 2 * np.pi, len(time))) * 5 + 15
    df = pd.DataFrame({"timestamp": time, "temperature": data1, "humidity": data2})
    df.set_index("timestamp", inplace=True)
    df.to_csv("sample_data.csv")
    return df


def run_example(log_level=logging.INFO, with_plots: bool = True, extrapolation: str = "constant"):
    # Set the log-level
    logging.basicConfig(level=log_level)

    # Create sample data
    original_df = create_sample_data()

    # Configure the agent
    agent_config = {
        "id": "DataSourceAgent",
        "modules": [
            {
                "module_id": "DataSource",
                "type": "csv_data_source",
                "data": "sample_data.csv",
                "t_sample": 10,
                "outputs": [
                    {"name": "temperature", "shared": True},
                    {"name": "humidity", "shared": True},
                ],
                "extrapolation": extrapolation,
            },
            {"module_id": "Logger", "type": "AgentLogger"},
        ],
    }

    # Set up the environment and agent
    env_config = {"rt": False, "factor": 1, "t_sample": 1}
    env = ag.Environment(config=env_config)
    agent = ag.Agent(config=agent_config, env=env)

    # Run the simulation
    env.run(until=600 * 2)  # Run for 10 minutes (600 seconds)
    results = agent.get_results()

    if with_plots:
        plot(results, original_df)

    return {"DataSourceAgent": results}


def plot(results, original_df):
    # Get results from the AgentLogger
    logger_results = results["Logger"]
    # Convert original_df index to seconds from start
    original_df.index = (original_df.index - original_df.index[0]).total_seconds()
    # Plot the results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    # Temperature plot
    original_df["temperature"].plot(
        ax=ax1,
        label="Original Temperature",
        alpha=0.7,
        marker="o",
    )
    logger_results["temperature"].plot(
        ax=ax1,
        label="DataSource Temperature",
        style="--",
        drawstyle="steps-post",
    )
    ax1.set_title("Temperature Comparison")
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_xlabel("Time (seconds)")
    ax1.legend()
    # Humidity plot
    original_df["humidity"].plot(
        ax=ax2,
        label="Original Humidity",
        alpha=0.7,
        marker="o",
    )
    logger_results["humidity"].plot(
        ax=ax2,
        label="DataSource Humidity",
        style="--",
        drawstyle="steps-post",
    )
    ax2.set_title("Humidity Comparison")
    ax2.set_ylabel("Humidity (%)")
    ax2.set_xlabel("Time (seconds)")
    ax2.legend()
    plt.tight_layout()
    plt.show()
    return results


if __name__ == "__main__":
    run_example(log_level=logging.INFO)
