# Room Multi-Agent System (MAS) Example

This directory contains examples demonstrating how to implement a Multi-Agent System (MAS) for room temperature control using `AgentLib`. It showcases the orchestration of multiple agents (Sensors, Controllers, and Actuators) interacting with a simulation environment represented by a Functional Mock-up Unit (FMU).

## Example Scripts

The folder contains three distinct execution scripts, each highlighting different features of the library:

### 1. `room_mas.py`
**Primary Example: Heating Control Simulation**
This script configures and runs a complete MAS simulation using an FMU. It demonstrates two different control strategies for room temperature regulation:
*   **PID Controller Agent**: Regulates heating based on proportional-integral-derivative logic.
*   **Bang-Bang Controller Agent**: A simple on/off switch logic.

The script initializes agents for sensors, room models, and controllers, and simulates their interaction over a defined time period.

### 2. `t_sample_simulation_demonstrating.py`
**Feature Demo: Time Sampling**
This example focuses on the synchronization between the simulation physics and the agent communication. It demonstrates how to split and manage:
*   **Simulation Time Steps**: The frequency at which the FMU physics are calculated.
*   **Communication Time Steps**: The frequency at which agents exchange data.

### 3. `view_simulation_in_dashboard.py`
**Feature Demo: Visualization**
This script runs the simulation with an emphasis on real-time or post-processing visualization. It shows how to connect the agent outputs to a dashboard for monitoring temperature curves and control signals.

---

## FMI Support and Tested Tools

`AgentLib` utilizes the FMI (Functional Mock-up Interface) standard to couple agent-based control logic with physical system models.

### Tested FMUs and Tools
This example and the underlying `AgentLib` library have been tested with FMUs generated from Dymola (Dassault Systèmes). The FMI co-simulation interface is implemented using the [FMPy](https://github.com/CATIA-Systems/FMPy) library, ensuring full compliance with the FMI 2.0 standard.

### FMI Standards
*   **FMI Version**: 2.0
*   **Type**: Co-Simulation (CS)

## Directory Structure

*   **`configs/`**: JSON configuration files defining agent parameters, environment settings, and communication setups.
*   **`models/`**: Contains the simulation models (e.g., `SimpleRoom.fmu`).
*   **`data/`**: Contains weather data used in the simulation.
*   **`results/`**: (Generated upon run) Stores simulation output data.

## How to Run

To execute the primary example, ensure `AgentLib` is installed and run:

```bash
python room_mas.py
