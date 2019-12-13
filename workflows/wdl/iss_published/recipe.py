import starfish
from starfish import FieldOfView
from starfish.image import Filter
from starfish.image import ApplyTransform, LearnTransform
from starfish.spots import FindSpots, DecodeSpots
from starfish.types import Axes, FunctionSource


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

    print(f"loading fov: {fov_str}")
    fov = experiment[fov_str]

    # note the structure of the 5D tensor containing the raw imaging data
    imgs = fov.get_image(FieldOfView.PRIMARY_IMAGES)
    dots = fov.get_image("dots")
    nuclei = fov.get_image("nuclei")

    print("Learning Transform")
    learn_translation = LearnTransform.Translation(reference_stack=dots, axes=Axes.ROUND, upsampling=1000)
    transforms_list = learn_translation.run(imgs.reduce({Axes.CH, Axes.ZPLANE}, func="max"))

    print("Applying transform")
    warp = ApplyTransform.Warp()
    registered_imgs = warp.run(imgs, transforms_list=transforms_list, in_place=True, verbose=True)

    print("Filter WhiteTophat")
    filt = Filter.WhiteTophat(masking_radius=15, is_volume=False)

    filtered_imgs = filt.run(registered_imgs, verbose=True, in_place=True)
    filt.run(dots, verbose=True, in_place=True)
    filt.run(nuclei, verbose=True, in_place=True)

    print("Detecting")
    detector = FindSpots.BlobDetector(
        min_sigma=1,
        max_sigma=10,
        num_sigma=30,
        threshold=0.01,
        measurement_type='mean',
    )
    dots_max_projector = Filter.Reduce((Axes.ROUND, Axes.ZPLANE), func=FunctionSource.np("max"))
    dots_max = dots_max_projector.run(dots)

    spots = detector.run(image_stack=filtered_imgs, reference_image=dots_max)

    decoder = DecodeSpots.PerRoundMaxChannel(codebook=experiment.codebook)
    decoded = decoder.run(spots=spots)
    df = decoded.to_decoded_dataframe()
    return df
