import unittest

from agentlib import Agent, Environment, AgentVariable
from agentlib.modules.communicator.local import LocalClient

default_data = {
    "name": "testvar",
    "type": "float",
    "value": 100,
    "ub": 200,
    "allowed_values": [100, 150],
    "shared": True,
    "unit": "testUnit",
    "description": "My Doc",
    "clip": True,
    "rdf_class": "MyRDFClass",
    "source": {"agent_id": "Send"}
}


class TestCommunicator(unittest.TestCase):

    def setUp(self) -> None:
        self.test_config = {
            "type": "local_broadcast",
            "module_id": "comm_module",
            "parse_json": True,
        }
        self.agent_send = Agent(config={"id": "Send", "modules": []}, env=Environment())
        self.agent_rec = Agent(config={"id": "Rec", "modules": []}, env=Environment())

    @unittest.skip
    def test_roundtrip_variable(self):
        variable = AgentVariable(**default_data)
        comm = LocalClient(config=self.test_config, agent=self.agent_send)
        payload = comm.short_dict(variable)
        var_json = comm.to_json(payload)
        variable2 = AgentVariable.from_json(var_json)
        comm_fields = ["alias", "source", "type", "timestamp", "value"]
        for field in comm_fields:
            self.assertEqual(variable.__getattribute__(field), variable2.__getattribute__(field))


if __name__ == '__main__':
    unittest.main()

