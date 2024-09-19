"""View the room_mas simulation in a dashboard.

To test the dashboard functionality, make sure you run the room_mas long enough
(either using real-time simulation, or a very long simulation duration). Also,
you need the optional dependency 'interactive'.

Make sure the simulation is already running and saving to a csv file!


"""

import os
from pathlib import Path

from agentlib.utils import simulator_dashboard

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    simulator_dashboard("res_room1.csv")  # accepts multiple files as *args
