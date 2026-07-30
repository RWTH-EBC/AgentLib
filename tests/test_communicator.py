import unittest
import tempfile
import os
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

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

    # NOTE: two known bugs were found while writing these tests and are intentionally
    # NOT exercised or fixed here (left for a separate follow-up):
    #   1. LocalCommunicator._receive_direct_callback (used when
    #      use_direct_callback_databroker=True) bypasses _handle_received_variable,
    #      so received-message logging silently never fires on that code path.
    #   2. CommunicationLogger.get_results()/get_results_incremental() for "detail"
    #      level raise FileNotFoundError if nothing was ever sent/received, since the
    #      log file is only created on first flush.

    def test_logging_none_incremental(self):
        """Test that 'none' log level's incremental fetch always returns (None, None)"""
        config = {**self.test_config, "communication_log_level": "none"}
        comm = LocalClient(config=config, agent=self.agent_send)

        variable = AgentVariable(**default_data)
        comm._send_only_shared_variables(variable)
        comm._handle_received_variable(variable, remote_agent_id="RemoteAgent")

        results, token = comm.get_results_incremental(update_token=None)
        self.assertIsNone(results)
        self.assertIsNone(token)

        results, token = comm.get_results_incremental(update_token="anything")
        self.assertIsNone(results)
        self.assertIsNone(token)

    def test_logging_basic_incremental(self):
        """Test incremental result retrieval for basic (counter) logging"""
        config = {**self.test_config, "communication_log_level": "basic"}
        comm = LocalClient(config=config, agent=self.agent_send)

        var1 = AgentVariable(**{**default_data, "name": "var1", "alias": "alias1"})
        comm._send_only_shared_variables(var1)

        # Initial call returns the current counts and a token
        results1, token1 = comm.get_results_incremental(update_token=None)
        self.assertIsInstance(results1, dict)
        self.assertEqual(results1["sent_counts"]["alias1"], 1)
        self.assertIsNotNone(token1)

        # Calling again with the up-to-date token before anything changed: no update
        results_unchanged, token_unchanged = comm.get_results_incremental(
            update_token=token1
        )
        self.assertIsNone(results_unchanged)
        self.assertEqual(token_unchanged, token1)

        # After another send, the stale token should yield fresh counts
        comm._send_only_shared_variables(var1)
        results2, token2 = comm.get_results_incremental(update_token=token1)
        self.assertIsInstance(results2, dict)
        self.assertEqual(results2["sent_counts"]["alias1"], 2)
        self.assertNotEqual(token2, token1)

    def test_cleanup_results_noop_for_none_and_basic(self):
        """Test that cleanup_results() is a no-op for 'none' and 'basic' levels"""
        for level in ["none", "basic"]:
            config = {**self.test_config, "communication_log_level": level}
            comm = LocalClient(config=config, agent=self.agent_send)

            variable = AgentVariable(**default_data)
            comm._send_only_shared_variables(variable)

            before = comm.get_results()
            comm.cleanup_results()  # should not raise
            after = comm.get_results()
            self.assertEqual(before, after)

    def test_communication_log_overwrite(self):
        """Test that a pre-existing log file is removed when overwrite is enabled"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test_overwrite.jsonl")
            with open(log_file, "w", encoding="utf-8") as f:
                f.write('{"stale": "entry"}\n')

            config = {
                **self.test_config,
                "communication_log_level": "detail",
                "communication_log_file": log_file,
                "communication_log_overwrite": True,
            }
            comm = LocalClient(config=config, agent=self.agent_send)

            variable = AgentVariable(**default_data)
            comm._send_only_shared_variables(variable)

            results = comm.get_results()
            self.assertEqual(len(results), 1)
            self.assertNotIn("stale", results.columns)

            comm.cleanup_results()

    def test_communication_log_default_path(self):
        """Test that omitting communication_log_file falls back to the default path"""
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)

                config = {**self.test_config, "communication_log_level": "detail"}
                comm = LocalClient(config=config, agent=self.agent_send)

                variable = AgentVariable(**default_data)
                comm._send_only_shared_variables(variable)
                comm.terminate()

                expected_path = Path("communicator_logs") / "Send_comm_module.jsonl"
                self.assertTrue(expected_path.exists())
            finally:
                os.chdir(cwd)

    def test_terminate_flushes_detail_log(self):
        """Test that terminate() flushes pending detail log entries to disk"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test_terminate.jsonl")
            config = {
                **self.test_config,
                "communication_log_level": "detail",
                "communication_log_file": log_file,
            }
            comm = LocalClient(config=config, agent=self.agent_send)

            variable = AgentVariable(**default_data)
            comm._send_only_shared_variables(variable)

            # Nothing flushed to disk yet
            self.assertFalse(Path(log_file).exists())

            comm.terminate()

            self.assertTrue(Path(log_file).exists())
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 1)

    def test_invalid_communication_log_level(self):
        """Test that an invalid communication_log_level is rejected by config validation"""
        config = {**self.test_config, "communication_log_level": "verbose"}
        with self.assertRaises(ValidationError):
            LocalClient(config=config, agent=self.agent_send)

    def test_visualize_results_none(self):
        """Test that visualize_results handles a None results payload"""
        try:
            from dash import html
        except ImportError:
            self.skipTest("dash not installed")

        result = LocalClient.visualize_results(
            None, module_id="comm_module", agent_id="Send"
        )
        self.assertIsInstance(result, html.Div)

    def test_visualize_results_basic(self):
        """Test that visualize_results renders basic (counter) results"""
        try:
            from dash import html
        except ImportError:
            self.skipTest("dash not installed")

        config = {**self.test_config, "communication_log_level": "basic"}
        comm = LocalClient(config=config, agent=self.agent_send)
        variable = AgentVariable(**default_data)
        comm._send_only_shared_variables(variable)
        comm._handle_received_variable(variable, remote_agent_id="RemoteAgent")

        result = LocalClient.visualize_results(
            comm.get_results(), module_id="comm_module", agent_id="Send"
        )
        self.assertIsInstance(result, html.Div)

    def test_visualize_results_detail(self):
        """Test that visualize_results renders detail (timeline) results"""
        try:
            from dash import html
        except ImportError:
            self.skipTest("dash not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test_visualize_detail.jsonl")
            config = {
                **self.test_config,
                "communication_log_level": "detail",
                "communication_log_file": log_file,
            }
            comm = LocalClient(config=config, agent=self.agent_send)
            variable = AgentVariable(**default_data)
            comm._send_only_shared_variables(variable)
            comm._handle_received_variable(variable, remote_agent_id="RemoteAgent")

            result = LocalClient.visualize_results(
                comm.get_results(), module_id="comm_module", agent_id="Send"
            )
            self.assertIsInstance(result, html.Div)

            comm.cleanup_results()

    def test_visualize_results_empty_detail(self):
        """Test that visualize_results handles an empty detail DataFrame"""
        try:
            from dash import html
        except ImportError:
            self.skipTest("dash not installed")

        result = LocalClient.visualize_results(
            pd.DataFrame(), module_id="comm_module", agent_id="Send"
        )
        self.assertIsInstance(result, html.Div)

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