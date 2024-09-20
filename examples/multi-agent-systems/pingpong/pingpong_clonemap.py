"""
This module implements a clonemap agent which starts the agentlib agent.

Before execution, be sure to:
1. Build the docker (from the agentlib-path):
docker build -f .\examples\multi-agent-systems\pingpong\Dockerfile -t agentlibtest .
2. Execute clonemap (using docker-compose up). See clonemaps doc for info on this.
"""

import json
import requests
from pathlib import Path


if __name__ == "__main__":
    import clonemapy.ams as ams

    URL = "http://localhost:30009/api/clonemap/mas"
    CFG_PATH = Path(__file__).parent.joinpath("clonemap_config.json")
    with open(CFG_PATH, "r") as file:
        DATA = json.load(file)
    requests.post(URL, json=DATA)
    ams.new_agent(
        host="localhost:30009",
        image="agentlibtest",
        secret="",
        masid=0,
        name="FirstAgent",
        custom='{"id":"FirstAgent", "modules":[{"module_id":"AgCom", "type":"clonemap", "subscriptions":["SecondAgent"], "log_level": "DEBUG"}, {"module_id":"Ping", "type":{"file": "pingpong_module.py", "class_name": "PingPong"}, "start":true, "initial_wait":10}]}',
    )
    ams.new_agent(
        host="localhost:30009",
        image="agentlibtest",
        secret="",
        masid=0,
        name="SecondAgent",
        custom='{"id":"SecondAgent", "modules":[{"module_id":"Ag2Com", "type":"clonemap", "subscriptions":["FirstAgent"], "log_level": "DEBUG"}, {"module_id":"Pong", "type":{"file": "pingpong_module.py", "class_name": "PingPong"}}]}',
    )
