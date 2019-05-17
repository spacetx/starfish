import starfish
from starfish import FieldOfView
from starfish.image import Filter
from starfish.image import ApplyTransform, LearnTransform
from starfish.spots import DetectSpots
from starfish.types import Axes


def process_fov(field_of_view, experiement_str):
    fov_str: str = f"fov_{int(field_of_view):03d}"

    # load experiment
    experiment = starfish.Experiment.from_json(experiement_str)

    print(f"loading fov: {fov_str}")
    fov = experiment[fov_str]

    # note the structure of the 5D tensor containing the raw imaging data
    imgs = fov.get_image(FieldOfView.PRIMARY_IMAGES)
    dots = fov.get_image("dots")
    nuclei = fov.get_image("nuclei")

    masking_radius = 15
    print("Filter WhiteTophat")
    filt = Filter.WhiteTophat(masking_radius, is_volume=False)

    filtered_imgs = filt.run(imgs, verbose=True, in_place=False)
    filt.run(dots, verbose=True, in_place=True)
    filt.run(nuclei, verbose=True, in_place=True)

    print("Learning Transform")
    learn_translation = LearnTransform.Translation(reference_stack=dots, axes=Axes.ROUND, upsampling=1000)
    transforms_list = learn_translation.run(imgs.max_proj(Axes.CH, Axes.ZPLANE))

    print("Applying transform")
    warp = ApplyTransform.Warp()
    registered_imgs = warp.run(filtered_imgs, transforms_list=transforms_list, in_place=False, verbose=True)

    print("Detecting")
    p = DetectSpots.BlobDetector(
        min_sigma=1,
        max_sigma=10,
        num_sigma=30,
        threshold=0.01,
        measurement_type='mean',
    )

    intensities = p.run(registered_imgs, blobs_image=dots, blobs_axes=(Axes.ROUND, Axes.ZPLANE))

    decoded = experiment.codebook.decode_per_round_max(intensities)
    df = decoded.to_decoded_spots()
    return df