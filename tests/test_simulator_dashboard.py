import unittest
from pathlib import Path

import pandas as pd
from dash import html, dcc

from agentlib.utils.plotting.simulator_dashboard import (
    load_new_data,
    update_data,
    format_time_axis,
    create_plot,
)


class TestSimulatorDashboard(unittest.TestCase):

    def test_load_new_data(self):
        # Create a temporary CSV file with a triple index column
        test_data = pd.DataFrame(
            {("output", "Variable1", "Real"): [10, 20, 30]}, index=[0, 1, 2]
        )
        # test_data.index.name = "time"

        temp_file = Path("test_data.csv")
        test_data.to_csv(temp_file)

        # Test initial load
        result = load_new_data(temp_file)
        expected_data = test_data.copy()
        expected_data.columns = expected_data.columns.droplevel(2)
        pd.testing.assert_frame_equal(result, expected_data)

        # Append new data to the CSV file
        new_data = pd.DataFrame({1: [40, 50]}, index=[3, 4])
        new_data.index.name = 0
        new_data.to_csv(temp_file, mode="a", header=False)

        # Test loading only the new data
        result = load_new_data(temp_file)
        expected_new_data = new_data.copy()
        pd.testing.assert_frame_equal(result, expected_new_data)

        # Clean up
        temp_file.unlink()

    def test_update_data(self):
        # Create existing data with a triple index column
        existing_data = pd.DataFrame(
            {("Output", "Variable1", "type"): [10, 20, 30]}, index=[0, 1, 2]
        )
        existing_data.index.name = "time"
        existing_data.columns = existing_data.columns.droplevel(2)

        # Create new data
        new_data = pd.DataFrame(
            {("Output", "Variable1", "type"): [40, 50]}, index=[3, 4]
        )
        new_data.index.name = "time"
        new_data.columns = new_data.columns.droplevel(2)

        # Test updating with new data
        result = update_data(existing_data, new_data)
        expected = pd.DataFrame(
            {("Output", "Variable1"): [10, 20, 30, 40, 50]}, index=[0, 1, 2, 3, 4]
        )
        expected.index.name = "time"
        pd.testing.assert_frame_equal(result, expected)

        # Test updating with empty existing data
        empty_existing_data = pd.DataFrame()
        result = update_data(empty_existing_data, new_data)
        pd.testing.assert_frame_equal(result, new_data)

    def test_format_time_axis(self):
        self.assertEqual(format_time_axis(30), (30, "s", "{:.0f}"))
        self.assertEqual(format_time_axis(300), (5, "min", "{:.1f}"))
        self.assertEqual(format_time_axis(18000), (5, "h", "{:.1f}"))

    def test_create_plot(self):
        df = pd.Series([10, 20, 30], index=[0, 1, 2], name="test")
        plot = create_plot(df, "Test Plot", "test-plot")
        self.assertIsInstance(plot, html.Div)
        self.assertEqual(len(plot.children), 1)  # only the plot
        self.assertIsInstance(plot.children[0], dcc.Graph)


if __name__ == "__main__":
    unittest.main()
