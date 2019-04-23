import os
import shutil
from pathlib import Path
from typing import Iterable, Optional

from starfish.core.util import exec


class RecipeTest:
    """This is a base class for testing recipes.  Each recipe test should define its recipe file,
    the input files, the output files, and a test method that verifies the correctness of the
    results.
    """
    @property
    def recipe(self) -> Path:
        raise NotImplementedError()

    @property
    def input_url_or_paths(self) -> Iterable[str]:
        raise NotImplementedError()

    @property
    def output_paths(self) -> Iterable[Path]:
        raise NotImplementedError()

    def verify_results(self, tempdir: Path):
        raise NotImplementedError()

    def test_run_recipe(self):
        cmdline = ["starfish", "recipe", "--recipe", self.recipe]
        for input_url_or_path in self.input_url_or_paths:
            cmdline.extend(["--input", input_url_or_path])
        for output_path in self.output_paths:
            cmdline.extend([
                "--output",
                lambda tempdir, *args, **kwargs: os.path.join(tempdir, os.fspath(output_path))])

        tempdir: Optional[str] = None
        try:
            tempdir = exec.stages([cmdline], keep_data=True)

            self.verify_results(Path(tempdir))
        finally:
            if tempdir is not None and os.getenv("TEST_KEEP_DATA") is None:
                shutil.rmtree(tempdir)
