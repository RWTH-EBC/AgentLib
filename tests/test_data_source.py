import os
import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from agentlib import Agent, Environment
from agentlib.modules.simulation.csv_data_source import (
    CSVDataSource,
    CSVDataSourceConfig,
)


class TestCSVDataSource(unittest.TestCase):

    def setUp(self):
        self.date_today = datetime.now()
        self.time = pd.date_range(
            self.date_today, self.date_today + timedelta(minutes=5), freq="min"
        )
        self.data1 = np.random.randint(1, high=100, size=len(self.time)) / 10
        self.data2 = np.random.randint(1, high=100, size=len(self.time)) / 10
        self.df = pd.DataFrame(
            {"index": self.time, "col1": self.data1, "col2": self.data2}
        )
        self.df.set_index("index", inplace=True)
        self.df.to_csv("test_df.csv")

        self.env_config = {"rt": False, "factor": 1}
        self.agent_config = {"id": "test_agent", "modules": []}
        self.env = Environment(config=self.env_config)
        self.agent = Agent(env=self.env, config=self.agent_config)

    def create_test_df(self, index_type="datetime"):
        data = {"col1": np.random.rand(5), "col2": np.random.rand(5)}

        if index_type == "datetime":
            index = pd.date_range(datetime.now(), periods=5, freq="s")
        elif index_type == "float":
            index = np.arange(5, dtype=float)
        elif index_type == "string_numbers":
            index = [str(i) for i in range(5)]
        elif index_type == "non_convertible":
            index = ["a", "b", "c", "d", "e"]

        df = pd.DataFrame(data, index=index)
        return df

    def test_datetime_index(self):
        original_df = self.create_test_df("datetime")
        config = self.get_data_source_config()
        config["data"] = original_df
        data_source = CSVDataSource(config=config, agent=self.agent)
        self.assertIsInstance(data_source.config.data.index, pd.Index)
        self.assertTrue(np.allclose(data_source.config.data.index, np.arange(5)))

    def test_float_index(self):
        original_df = self.create_test_df("float")
        config = self.get_data_source_config()
        config["data"] = original_df
        data_source = CSVDataSource(config=config, agent=self.agent)
        self.assertIsInstance(data_source.config.data.index, pd.Index)
        self.assertTrue(np.allclose(data_source.config.data.index, np.arange(5)))

    def test_string_number_index(self):
        original_df = self.create_test_df("string_numbers")
        config = self.get_data_source_config()
        config["data"] = original_df
        data_source = CSVDataSource(config=config, agent=self.agent)
        self.assertIsInstance(data_source.config.data.index, pd.Index)
        self.assertTrue(np.allclose(data_source.config.data.index, np.arange(5)))

    def test_non_convertible_index(self):
        original_df = self.create_test_df("non_convertible")
        config = self.get_data_source_config()
        config["data"] = original_df
        with self.assertRaises(ValueError):
            CSVDataSource(config=config, agent=self.agent)

    def get_data_source_config(self, **kwargs):
        return {
            "module_id": "Test_Data_Source",
            "type": {"file": "csv_data_source.py", "class_name": "CSVDataSource"},
            "data": kwargs.get("data", "test_df.csv"),
            "t_sample": kwargs.get("t_sample", 1),
            "data_offset": kwargs.get("data_offset", 0),
            "outputs": kwargs.get("outputs", []),
        }

    def test_data_source_initialization(self):
        config = self.get_data_source_config()
        data_source = CSVDataSource(config=config, agent=self.agent)
        self.assertIsInstance(data_source.config, CSVDataSourceConfig)
        self.assertEqual(data_source.config.t_sample, 1)
        self.assertEqual(data_source.config.data_offset, 0)

    def test_data_loading(self):
        config = self.get_data_source_config()
        data_source = CSVDataSource(config=config, agent=self.agent)
        self.assertIsInstance(data_source.config.data, pd.DataFrame)
        self.assertEqual(len(data_source.config.data), len(self.df))

    def test_data_offset(self):
        offset = 60
        config = self.get_data_source_config(data_offset=offset)
        data_source = CSVDataSource(config=config, agent=self.agent)
        self.assertEqual(data_source.config.data.index[0], -offset)

    def test_interpolation(self):
        config = self.get_data_source_config(t_sample=30)  # 30 seconds
        data_source = CSVDataSource(config=config, agent=self.agent)
        self.assertEqual(
            len(data_source.data_tuples), 11
        )  # 5 minutes = 300 seconds, 300/30 + 1 = 11

    def test_data_exhaustion(self):
        config = self.get_data_source_config()
        data_source = CSVDataSource(config=config, agent=self.agent)
        for _ in data_source.data_tuples:
            data_source._get_next_data()
        last_data = data_source._get_next_data()
        self.assertEqual(last_data, data_source.data_tuples[-1])

    def tearDown(self):
        if os.path.exists("test_df.csv"):
            os.remove("test_df.csv")


if __name__ == "__main__":
    unittest.main()
