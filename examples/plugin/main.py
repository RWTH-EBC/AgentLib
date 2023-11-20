"""Example file for loading plugins."""

from agentlib.utils.multi_agent_system import LocalMASAgency


agent_configs = [
    {
        "id": "FirstAgent",
        "modules": [
            {"module_id": "Ag1Com", "type": "local_broadcast"},
            {"module_id": "Ping", "type": "my_plugin.new_module"},
        ],
    },
    {
        "id": "SecondAgent",
        "modules": [
            {"module_id": "Ag2Com", "type": "local_broadcast"},
            {"module_id": "Pong", "type": "my_plugin.new_module2"},
        ],
    },
]


def main():
    mas = LocalMASAgency(agent_configs=agent_configs)
    mas.run(until=30)


if __name__ == "__main__":
    main()
