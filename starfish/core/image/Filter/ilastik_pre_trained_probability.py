import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import scipy.ndimage as ndi

from starfish.core.imagestack.imagestack import ImageStack
from ._base import FilterAlgorithm


class IlastikPretrainedProbability(FilterAlgorithm):
    """
    Use an existing ilastik pixel classifier to generate a probability image for a dapi image.
    NOTE: This api may not be used without a downloaded and installed version of Ilastik. Visit
    https://www.ilastik.org/download.html to download.

    Parameters
    ----------
    ilastik_executable: Union[Path, str]
        Path to run_ilastik.sh (Linux and Mac) or ilastik.bat (Windows) needed for running
        ilastik in headless mode. Typically the script is located: "/Applications/{
        ILASTIK_VERSION}/Contents/ilastik-release/run_ilastik.sh"

    ilastik_project: Union[Path, str]
        path to ilastik project .ilp file

    """

    def __init__(self, ilastik_executable: Union[Path, str], ilastik_project: Union[Path, str]):
        ilastik_executable = Path(ilastik_executable)
        if not ilastik_executable.exists():
            raise EnvironmentError("Can not find run_ilastik.sh or ilastik.bat. Make sure you've "
                                   "provided the correct location. If you need to download ilastik "
                                   "please visit: https://www.ilastik.org/download.html")
        self.ilastik_executable = ilastik_executable
        self.ilastik_project = ilastik_project

    def run(
            self,
            stack: ImageStack,
            in_place: bool = False,
            verbose: bool = False,
            n_processes: Optional[int] = None,
            *args,
    ) -> Optional[ImageStack]:
        """
        Use a pre trained probability pixel classification model to generate probabilities
        for a dapi image

        Parameters
        ----------
        stack : ImageStack
            Dapi image to be run through ilastik.
        in_place : bool
            This parameter is ignored for this filter.
        verbose : bool
            This parameter is ignored for this filter.
        n_processes : Optional[int]
            This parameter is ignored for this filter.

        Returns
        -------
        ImageStack :
            A new ImageStack created from the cell probabilities provided by ilastik.
        """
        if stack.num_rounds != 1:
            raise ValueError(
                f"{IlastikPretrainedProbability.__name__} given an image with more than one round "
                f"{stack.num_rounds}")
        if stack.num_chs != 1:
            raise ValueError(
                f"{IlastikPretrainedProbability.__name__} given an image with more than one "
                f"channel "
                f"{stack.num_chs}")

        # temp files
        with tempfile.TemporaryDirectory() as temp_dir:
            dapi_file = f"{temp_dir}_dapi.npy"
            output_file = f"{temp_dir}_dapi_Probabilities.h5"
            np.save(dapi_file, stack.xarray.values.squeeze())

            # env {} is needed to fix the weird virtualenv stuff
            subprocess.check_call(
                [self.ilastik_executable,
                 '--headless',
                 '--project',
                 self.ilastik_project,
                 "--output_filename_format",
                 output_file,
                 dapi_file], env={})  # type: ignore

        return self.import_ilastik_probabilities(output_file)

    @classmethod
    def import_ilastik_probabilities(
            cls,
            path_to_h5_file: Union[str, Path],
            dataset_name: str = "exported_data"
    ) -> ImageStack:
        """
        Import cell probabilities provided by ilastik as an ImageStack.

        Parameters
        ----------
        path_to_h5_file : Union[str, Path]
            Path to the .h5 file outputted by ilastik
        dataset_name : str
            Name of dataset in ilastik Export Image Settings
        Returns
        -------
        ImageStack :
            A new ImageStack created from the cell probabilities provided by ilastik.
        """

        h5 = h5py.File(path_to_h5_file)
        probability_images = h5[dataset_name][:]
        h5.close()
        cell_probabilities, _ = probability_images[:, :, 0], probability_images[:, :, 1]
        label_array = ndi.label(cell_probabilities)[0]
        label_array = label_array[np.newaxis, np.newaxis, np.newaxis, ...]
        return ImageStack.from_numpy(label_array)
