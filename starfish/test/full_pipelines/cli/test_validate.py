import unittest

from starfish.util import exec


class TestValidateCommand(unittest.TestCase):

    STAGES = (
        [
            "starfish", "validate", "--help"
        ],
    )

    def test_run_pipeline(self):
        exec.stages(TestValidateCommand.STAGES)
