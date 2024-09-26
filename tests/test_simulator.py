import os
import shutil
import unittest
import uuid

import numpy as np
import pandas as pd
from pydantic import ValidationError

from agentlib.core import (
    Agent,
    Environment,
    Model,
    ModelConfig,
    ModelOutputs,
    ModelOutput,
    ModelInput,
    ModelInputs,
    datamodels,
)
from agentlib.modules.simulation.simulator import Simulator


# pylint: disable=missing-module-docstring,missing-class-docstring


class DummyModelConfig(ModelConfig):
    outputs: ModelOutputs = [
        ModelOutput(name="output", value=0),
        ModelOutput(name="output_2", value=0),
    ]
    inputs: ModelInputs = [ModelInput(name="input", value=0)]
    states: datamodels.ModelStates = [datamodels.ModelState(name="local", value=10)]
    parameters: datamodels.ModelParameters = [
        datamodels.ModelParameter(name="par", value=10)
    ]


class DummyModel(Model):
    config: DummyModelConfig

    def do_step(self, *, t_start, t_sample=None):
        pass

    def initialize(self, **kwargs):
        pass


class TestSimulator(unittest.TestCase):
    def setUp(self) -> None:
        self.t_start = np.random.randint(2)
        self.t_stop = np.random.randint(self.t_start + 10, high=self.t_start + 10000)
        self.t_sample = np.random.randint(1, high=self.t_stop - self.t_start)

        # Test model injection:
        self.workdir = os.path.join(os.getcwd(), "_temp_simulator_test")
        os.makedirs(self.workdir, exist_ok=True)
        self.model_config = {"type": {"file": __file__, "class_name": "DummyModel"}}

    def get_agent(self):
        env_config = {"rt": False, "factor": 1}
        agent_config = {"id": "TestAgent", "modules": []}
        env = Environment(config=env_config)
        return Agent(env=env, config=agent_config)

    def _get_module_cfg(self, **kwargs):
        """Dynamically build config to avoid duplicate code:"""
        return {
            "module_id": "Simulator",
            "type": "simulator",
            "model": kwargs.get("model_config", self.model_config),
            "t_start": kwargs.get("t_start", self.t_start),
            "t_stop": kwargs.get("t_stop", self.t_stop),
            "t_sample": kwargs.get("t_sample", self.t_sample),
            "save_results": kwargs.get("save_results", True),
            "result_filename": kwargs.get(
                "result_filename", os.path.join(self.workdir, "temp_test_file.csv")
            ),
            "result_causalities": kwargs.get(
                "result_causalities", ["input", "parameter", "local", "output"]
            ),
            "measurement_uncertainty": 0.01,
            "inputs": [{"name": "input", "value": 10}],
            "outputs": [{"name": "output"}, {"name": "output_2"}],
        }

    def test_simulator_parameters(self):
        """Test if the parameters are correctly set and accessible."""
        simulator = Simulator(config=self._get_module_cfg(), agent=self.get_agent())
        self.assertEqual(simulator.config.t_start, self.t_start)
        self.assertEqual(simulator.config.t_sample, self.t_sample)
        self.assertEqual(simulator.config.t_stop, self.t_stop)

    def test_simulation(self):
        simulator = Simulator(config=self._get_module_cfg(), agent=self.get_agent())
        step = simulator.process()
        next(step)
        res = simulator.get_results()
        self.assertEqual(res.iloc[-1]["input"], 10)

    def test_inputs_and_pars(self):
        simulator = Simulator(config=self._get_module_cfg(), agent=self.get_agent())
        simulator._callback_update_model_input(
            datamodels.AgentVariable(name="asdsa", value=100), name="input"
        )
        simulator._callback_update_model_parameter(
            datamodels.AgentVariable(name="dasdsafsa", value=100), name="par"
        )
        simulator.run()
        res = simulator.get_results()
        self.assertEqual(res.iloc[-1]["input"], 100)
        self.assertEqual(res.iloc[-1]["par"], 100)

    def test_save_results(self):
        simulator = Simulator(config=self._get_module_cfg(), agent=self.get_agent())
        step = simulator.process()
        next(step)
        # With save_results = True
        self.assertIsInstance(simulator.get_results(), pd.DataFrame)
        self.assertTrue(os.path.isfile(simulator.config.result_filename))
        # Now without results:
        simulator = Simulator(
            config=self._get_module_cfg(save_results=False), agent=self.get_agent()
        )
        self.assertIsNone(simulator.get_results())
        # Without save:
        simulator = Simulator(
            config=self._get_module_cfg(save_results=False), agent=self.get_agent()
        )
        step = simulator.process()
        next(step)
        self.assertIsNone(simulator.get_results())
        # Without filename:
        simulator = Simulator(
            config=self._get_module_cfg(result_filename=None), agent=self.get_agent()
        )
        step = simulator.process()
        next(step)
        self.assertIsInstance(simulator.get_results(), pd.DataFrame)

    def test_wrong_causality(self):
        with self.assertRaises(ValidationError):
            Simulator(
                config=self._get_module_cfg(result_causalities=["notacausality"]),
                agent=self.get_agent(),
            )

    def test_no_files(self):
        simulator = Simulator(
            config=self._get_module_cfg(result_filename=None, save_results=True),
            agent=self.get_agent(),
        )
        self.assertIsNone(simulator.config.result_filename)

    def test_model_injection(self):
        """Test if model injection by file is possible."""
        # First check normal instance with class from this module:
        simulator = Simulator(
            config=self._get_module_cfg(result_filename=None, save_results=True),
            agent=self.get_agent(),
        )
        self.assertIsInstance(simulator.model, Model)

        # Now using a newly created python file
        custom_model_py = (
            "from agentlib.core import Model\n"
            "class DummyModel(Model):\n"
            "\tdef do_step(self, *, t_start, dt=None):\n"
            "\t\tpass\n"
            "\tdef initialize(self, **kwargs):\n"
            "\t\tpass"
        )
        fpath = os.path.join(self.workdir, f"{uuid.uuid4()}.py")
        with open(fpath, "w+") as file:
            file.write(custom_model_py)
        model_config = {"type": {"file": fpath, "class_name": "DummyModel"}}
        config = self._get_module_cfg(model_config=model_config)
        config["inputs"] = []
        simulator = Simulator(config=config, agent=self.get_agent())
        self.assertIsInstance(simulator.model, Model)

    def tearDown(self) -> None:
        """Remove files"""
        shutil.rmtree(self.workdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
