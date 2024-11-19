import os
import shutil
import unittest

import numpy as np
import pandas as pd

from agentlib.core import Agent, Environment
from agentlib.core.datamodels import AgentVariable
from agentlib.modules.utils.agent_logger import AgentLogger


class TestAgentLogger(unittest.TestCase):
    def setUp(self):
        """Set up test environment and working directory"""
        self.workdir = os.path.join(os.getcwd(), "_temp_logger_test")
        os.makedirs(self.workdir, exist_ok=True)

        # Set up random sampling time
        self.t_sample = np.random.randint(1, 1000)

    def get_agent(self):
        """Helper method to create a test agent"""
        env_config = {"rt": False, "factor": 1}
        agent_config = {"id": "TestAgent", "modules": []}
        env = Environment(config=env_config)
        return Agent(env=env, config=agent_config)

    def _get_module_cfg(self, **kwargs):
        """Helper method to create logger config"""
        return {
            "module_id": "Logger",
            "type": "agent_logger",
            "t_sample": kwargs.get("t_sample", self.t_sample),
            "values_only": kwargs.get("values_only", True),
            "clean_up": kwargs.get("clean_up", True),
            "overwrite_log": kwargs.get("overwrite_log", False),
            "filename": kwargs.get(
                "filename",
                os.path.join(self.workdir, "test_log.json")
            )
        }

    def test_logger_parameters(self):
        """Test if the parameters are correctly set and accessible"""
        logger = AgentLogger(config=self._get_module_cfg(), agent=self.get_agent())
        self.assertEqual(logger.config.t_sample, self.t_sample)
        self.assertTrue(logger.config.values_only)
        self.assertTrue(logger.config.clean_up)
        self.assertFalse(logger.config.overwrite_log)

    def test_logging_values(self):
        """Test if values are correctly logged"""
        logger = AgentLogger(config=self._get_module_cfg(), agent=self.get_agent())

        # Create test variable
        test_var = AgentVariable(name="test", value=42.0, alias="test_alias")

        # Log the variable
        logger._callback_values(test_var)
        logger._log()

        # Load and check results
        results = logger.get_results()
        self.assertIsInstance(results, pd.DataFrame)
        self.assertTrue("test_alias" in results.columns)

    def test_logging_full(self):
        """Test if full variable information is correctly logged"""
        logger = AgentLogger(
            config=self._get_module_cfg(values_only=False),
            agent=self.get_agent()
        )

        # Create test variable
        test_var = AgentVariable(name="test", value=42.0, alias="test_alias")

        # Log the variable
        logger._callback_full(test_var)
        logger._log()

        # Load and check results
        results = logger.get_results()
        self.assertIsInstance(results, pd.DataFrame)
        self.assertTrue("test_alias" in results.columns)

    def test_file_overwrite(self):
        """Test file overwrite behavior"""
        # Create initial logger and file
        config = self._get_module_cfg()
        logger = AgentLogger(config=config, agent=self.get_agent())

        # Write some data to create the file
        test_var = AgentVariable(name="test", value=42.0, alias="test_alias")
        logger._callback_values(test_var)
        logger._log()

        # Verify file exists
        self.assertTrue(os.path.exists(logger.filename))

        # Try to create another logger with same filename
        with self.assertRaises(FileExistsError):
            AgentLogger(config=config, agent=self.get_agent())

        # Test with overwrite enabled
        config["overwrite_log"] = True
        logger_new = AgentLogger(config=config, agent=self.get_agent())

        # Write new data to ensure the logger is functional
        test_var_new = AgentVariable(name="test2", value=43.0, alias="test_alias2")
        logger_new._callback_values(test_var_new)
        logger_new._log()
        self.assertTrue(os.path.exists(logger_new.filename))

        # Load and verify the data
        results = logger_new.get_results()
        self.assertIsInstance(results, pd.DataFrame)
        self.assertTrue("test_alias2" in results.columns)

    def test_cleanup(self):
        """Test cleanup functionality"""
        logger = AgentLogger(
            config=self._get_module_cfg(clean_up=True),
            agent=self.get_agent()
        )

        # Log something
        test_var = AgentVariable(name="test", value=42.0, alias="test_alias")
        logger._callback_values(test_var)
        logger._log()

        # Verify file exists
        self.assertTrue(os.path.exists(logger.filename))

        # Cleanup
        logger.cleanup_results()
        self.assertFalse(os.path.exists(logger.filename))

    def test_load_from_file(self):
        """Test loading results from file"""
        logger = AgentLogger(config=self._get_module_cfg(), agent=self.get_agent())

        # Log multiple variables
        test_vars = [
            AgentVariable(name="test1", value=42.0, alias="test_alias1"),
            AgentVariable(name="test2", value=43.0, alias="test_alias2")
        ]

        for var in test_vars:
            logger._callback_values(var)
        logger._log()

        # Test loading with different merge_sources options
        df_merged = AgentLogger.load_from_file(
            logger.filename,
            merge_sources=True
        )
        self.assertIsInstance(df_merged, pd.DataFrame)

        df_unmerged = AgentLogger.load_from_file(
            logger.filename,
            merge_sources=False
        )
        self.assertIsInstance(df_unmerged, pd.DataFrame)
        self.assertTrue(isinstance(df_unmerged.columns, pd.MultiIndex))

    def tearDown(self):
        """Clean up test files"""
        shutil.rmtree(self.workdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()