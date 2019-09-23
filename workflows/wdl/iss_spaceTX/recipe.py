import numpy as np
import starfish
from starfish.types import Axes


def process_fov(field_num: int, experiment_str: str):
    """Process a single field of view of ISS data
    Parameters
    ----------
    field_num : int
        the field of view to process
    experiment_str : int
        path of experiment json file

    Returns
    -------
    DecodedSpots :
        tabular object containing the locations of detected spots.
    """
    fov_str: str = f"fov_{int(field_num):03d}"

    # load experiment
    experiment = starfish.Experiment.from_json(experiment_str)

    fov = experiment[fov_str]
    imgs = fov.get_image(starfish.FieldOfView.PRIMARY_IMAGES)
    max_projector = starfish.image.Filter.Reduce((Axes.CH, Axes.ZPLANE))
    dots = max_projector.run(imgs)

    # filter
    filt = starfish.image.Filter.WhiteTophat(masking_radius=15, is_volume=False)
    filtered_imgs = filt.run(imgs, verbose=True, in_place=False)
    filt.run(dots, verbose=True, in_place=True)

    # find threshold
    tmp = dots.sel({Axes.ROUND:0, Axes.CH:0, Axes.ZPLANE:0})
    dots_threshold = np.percentile(np.ravel(tmp.xarray.values), 50)

    bd = starfish.spots.FindSpots.BlobDetector(
        min_sigma=1,
        max_sigma=10,
        num_sigma=30,
        threshold=dots_threshold,
        measurement_type='mean',
    )

    dots_max_projector = starfish.image.Filter.Reduce(
        (Axes.ROUND, Axes.ZPLANE), func="max",
        module=starfish.image.Filter.Reduce.FunctionSource.np)
    dots_max = dots_max_projector.run(dots)

    spots = bd.run(image_stack=filtered_imgs, reference_image=dots_max)

    decoder = starfish.spots.DecodeSpots.PerRoundMaxChannel(codebook=experiment.codebook)
    decoded = decoder.run(spots=spots)

    df = decoded.to_decoded_dataframe()
    return df
