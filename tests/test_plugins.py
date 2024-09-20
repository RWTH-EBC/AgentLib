# todo maybe use the example somehow? need to test model as well
import unittest


class TestPlugin(unittest.TestCase):
    """Class to test only the base-model."""

    def setUp(self) -> None:
        pass

    def test_model_plugin(self): ...

    def test_module_plugin(self): ...

    def test_module_not_found_error(self): ...
