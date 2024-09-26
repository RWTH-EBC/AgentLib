"""Module with functions to test the if examples are
executable."""

import unittest
import os
import subprocess
import logging
import os
import pathlib
import subprocess
import unittest

import pandas as pd

from agentlib.utils import custom_injection
from agentlib.utils.local_broadcast_broker import LocalBroadcastBroker


class TestExamples(unittest.TestCase):
    """Test all examples inside the agentlib"""

    def setUp(self) -> None:
        self.timeout = 10  # Seconds which the script is allowed to run
        self.main_cwd = os.getcwd()

    def tearDown(self) -> None:
        broker = LocalBroadcastBroker()
        broker.delete_all_clients()
        # Change back cwd:
        os.chdir(self.main_cwd)

    def _run_example(self, example, timeout=None):
        if timeout is None:
            timeout = self.timeout
        ex_py = (
            pathlib.Path(__file__).absolute().parents[1].joinpath("examples", example)
        )
        try:
            subprocess.check_output(
                ["python", ex_py], stderr=subprocess.STDOUT, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            pass
        except subprocess.CalledProcessError as proc_err:
            raise Exception(proc_err.output.decode("utf-8")) from proc_err

    def _run_example_with_return(self, file: str, func_name: str, **kwargs):
        file = pathlib.Path(__file__).absolute().parents[1].joinpath("examples", file)

        # Custom file import
        test_func = custom_injection({"file": file, "class_name": func_name})
        results = test_func(**kwargs)
        self.assertIsInstance(results, dict)
        agent_results = results.popitem()[1]
        self.assertIsInstance(agent_results, dict)
        val = agent_results.popitem()[1]
        self.assertIsInstance(val, pd.DataFrame)

    def test_room_mas(self):
        """Test the room_mas example"""
        import sys

        if "linux" in sys.platform:
            self.skipTest("FMU test not yet supported")
        self._run_example_with_return(
            file="multi-agent-systems//room_mas//room_mas.py",
            func_name="run_example",
            until=8640,
            with_plots=False,
            log_level=logging.DEBUG,
        )

    def test_pingpong(self):
        """Test the pingpong example using various communicators"""
        # self._run_example(example="multi-agent-systems//pingpong//pingpong_mqtt.py")
        self._run_example(
            example="multi-agent-systems//pingpong//pingpong_local_broadcast.py"
        )
        self._run_example(example="multi-agent-systems//pingpong//pingpong_local.py")

    def test_bangbang(self):
        """Test the bangbang example"""
        self._run_example_with_return(
            file="controller//bangbang_with_simulator.py",
            func_name="run_example",
            with_plots=False,
            log_level=logging.FATAL,
        )

    def test_pid(self):
        """Test the pid example"""
        self._run_example_with_return(
            file="controller//pid_with_simulator.py",
            func_name="run_example",
            with_plots=False,
            log_level=logging.FATAL,
        )

    def test_csv_data_source(self):
        """Test the pid example"""
        self._run_example_with_return(
            file="simulation//csv_data_source.py",
            func_name="run_example",
            with_plots=False,
            log_level=logging.FATAL,
            extrapolation="constant"
        )
        self._run_example_with_return(
            file="simulation//csv_data_source.py",
            func_name="run_example",
            with_plots=False,
            log_level=logging.FATAL,
            extrapolation="repeat"
        )
        self._run_example_with_return(
            file="simulation//csv_data_source.py",
            func_name="run_example",
            with_plots=False,
            log_level=logging.FATAL,
            extrapolation="backwards"
        )


    def test_scipy_model(self):
        """Tests the scipy model example"""
        self._run_example(example="models//scipy//scipy_example.py")


if __name__ == "__main__":
    unittest.main()
