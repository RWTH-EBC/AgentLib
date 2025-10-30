import unittest
import tempfile
import os
from pathlib import Path

import pandas as pd

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
    "source": {"agent_id": "Send"},
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

    def test_roundtrip_variable(self):
        variable = AgentVariable(**default_data)
        comm = LocalClient(config=self.test_config, agent=self.agent_send)
        payload = comm.short_dict(variable)
        var_json = comm.to_json(payload)
        variable2 = AgentVariable.from_json(var_json)
        comm_fields = ["alias", "source", "type", "timestamp", "value"]
        for field in comm_fields:
            self.assertEqual(
                variable.__getattribute__(field), variable2.__getattribute__(field)
            )

    def test_pd_series(self):
        """Tests whether pandas series are sent correctly"""
        data = {**default_data, "value": pd.Series({0: 1, 10: 2}), "type": "pd.Series"}
        variable = AgentVariable(**data)
        comm_parse = LocalClient(config=self.test_config, agent=self.agent_send)
        comm_no_parse = LocalClient(
            config={**self.test_config, "parse_json": False}, agent=self.agent_send
        )

        # communicator with json parsing
        payload = comm_parse.short_dict(variable)
        var_json = comm_parse.to_json(payload)
        variable2 = AgentVariable.from_json(var_json)
        pd.testing.assert_series_equal(variable.value, variable2.value)

        # communicator without json parsing
        payload = comm_no_parse.short_dict(variable)
        payload["name"] = payload["alias"]
        variable2 = AgentVariable(**payload)
        pd.testing.assert_series_equal(variable.value, variable2.value)

    def test_logging_none(self):
        """Test that 'none' log level produces no results"""
        config = {**self.test_config, "communication_log_level": "none"}
        comm = LocalClient(config=config, agent=self.agent_send)

        # Trigger some sends and receives
        variable = AgentVariable(**default_data)
        comm._send_only_shared_variables(variable)
        comm._handle_received_variable(variable, remote_agent_id="RemoteAgent")

        # Should return None
        results = comm.get_results()
        self.assertIsNone(results)

    def test_logging_basic(self):
        """Test that 'basic' log level produces count dictionaries"""
        config = {**self.test_config, "communication_log_level": "basic"}
        comm = LocalClient(config=config, agent=self.agent_send)

        # Send multiple variables with different aliases
        var1 = AgentVariable(**{**default_data, "name": "var1", "alias": "alias1"})
        var2 = AgentVariable(**{**default_data, "name": "var2", "alias": "alias2"})
        var3 = AgentVariable(**{**default_data, "name": "var1", "alias": "alias1"})

        comm._send_only_shared_variables(var1)
        comm._send_only_shared_variables(var2)
        comm._send_only_shared_variables(var3)  # alias1 sent twice

        # Receive variables from different sources
        comm._handle_received_variable(var1, remote_agent_id="Agent1")
        comm._handle_received_variable(var2, remote_agent_id="Agent2")
        comm._handle_received_variable(var1, remote_agent_id="Agent1")  # Agent1/alias1 twice

        results = comm.get_results()

        # Check structure
        self.assertIsInstance(results, dict)
        self.assertIn("sent_counts", results)
        self.assertIn("received_counts", results)

        # Check sent counts
        self.assertEqual(results["sent_counts"]["alias1"], 2)
        self.assertEqual(results["sent_counts"]["alias2"], 1)

        # Check received counts (keys are strings of tuples)
        received = results["received_counts"]
        self.assertEqual(received["('Agent1', 'alias1')"], 2)
        self.assertEqual(received["('Agent2', 'alias2')"], 1)

    def test_logging_detail(self):
        """Test that 'detail' log level produces a DataFrame with timeline data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test_comm.jsonl")
            config = {
                **self.test_config,
                "communication_log_level": "detail",
                "communication_log_file": log_file,
            }
            comm = LocalClient(config=config, agent=self.agent_send)

            # Send and receive some variables
            var1 = AgentVariable(**{**default_data, "name": "var1", "alias": "temp_sensor"})
            var2 = AgentVariable(**{**default_data, "name": "var2", "alias": "pressure_sensor"})

            comm._send_only_shared_variables(var1)
            comm._send_only_shared_variables(var2)
            comm._handle_received_variable(var1, remote_agent_id="SensorAgent1")
            comm._handle_received_variable(var2, remote_agent_id="SensorAgent2")

            # Get results
            results = comm.get_results()

            # Check it's a DataFrame
            self.assertIsInstance(results, pd.DataFrame)

            # Check structure
            self.assertEqual(len(results), 4)  # 2 sent + 2 received
            self.assertIn("timestamp", results.columns)
            self.assertIn("direction", results.columns)
            self.assertIn("alias", results.columns)
            self.assertIn("own_agent_id", results.columns)
            self.assertIn("remote_agent_id", results.columns)

            # Check sent messages
            sent_df = results[results["direction"] == "sent"]
            self.assertEqual(len(sent_df), 2)
            self.assertIn("temp_sensor", sent_df["alias"].values)
            self.assertIn("pressure_sensor", sent_df["alias"].values)

            # Check received messages
            received_df = results[results["direction"] == "received"]
            self.assertEqual(len(received_df), 2)
            self.assertIn("SensorAgent1", received_df["remote_agent_id"].values)
            self.assertIn("SensorAgent2", received_df["remote_agent_id"].values)

            # Cleanup
            comm.cleanup_results()
            self.assertFalse(Path(log_file).exists())

    def test_logging_detail_incremental(self):
        """Test incremental result retrieval for detail logging"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test_comm_incremental.jsonl")
            config = {
                **self.test_config,
                "communication_log_level": "detail",
                "communication_log_file": log_file,
            }
            comm = LocalClient(config=config, agent=self.agent_send)

            # Send first batch
            var1 = AgentVariable(**{**default_data, "name": "var1", "alias": "alias1"})
            comm._send_only_shared_variables(var1)

            # Get initial results
            results1, token1 = comm.get_results_incremental(update_token=None)
            self.assertIsInstance(results1, pd.DataFrame)
            self.assertEqual(len(results1), 1)
            self.assertIsNotNone(token1)

            # Send second batch
            var2 = AgentVariable(**{**default_data, "name": "var2", "alias": "alias2"})
            comm._send_only_shared_variables(var2)
            comm._handle_received_variable(var1, remote_agent_id="Agent1")

            # Get incremental results
            results2, token2 = comm.get_results_incremental(update_token=token1)
            self.assertIsInstance(results2, pd.DataFrame)
            self.assertEqual(len(results2), 2)  # 1 sent + 1 received
            self.assertEqual(token2, token1 + 2)

            # Cleanup
            comm.cleanup_results()

    def test_pd_series_no_json(self):
        """Tests whether pandas series are sent correctly"""
        data = {**default_data, "value": pd.Series({0: 1, 10: 2}), "type": "pd.Series"}
        variable = AgentVariable(**data)
        _config = self.test_config.copy()
        _config["parse_json"] = False
        comm = LocalClient(config=_config, agent=self.agent_send)
        payload = comm.short_dict(variable)
        pd.testing.assert_series_equal(variable.value, payload["value"])


if __name__ == "__main__":
    unittest.main()