import click


@click.command()
@click.option('--experiment', type=str, help='experiment URL')
@click.option('--fov', type=int, help='field of view number')
def test_experiment(experiment, fov):

    import numpy as np
    import starfish
    from starfish.types import Axes
    fov_str: str = f"fov_{int(fov):03d}"

    # load experiment
    experiment = starfish.Experiment.from_json(experiment)

    fov = experiment[fov_str]
    imgs = fov.get_image(starfish.FieldOfView.PRIMARY_IMAGES)
    dots = imgs.max_proj(Axes.CH)

    # filter
    filt = starfish.image.Filter.WhiteTophat(masking_radius=15, is_volume=False)
    filtered_imgs = filt.run(imgs, verbose=True, in_place=False)
    filt.run(dots, verbose=True, in_place=True)

    # find threshold
    tmp = dots.sel({Axes.ROUND:0, Axes.CH:0, Axes.ZPLANE:0})
    dots_threshold = np.percentile(np.ravel(tmp.xarray.values), 50)

    # find spots
    p = starfish.spots.DetectSpots.BlobDetector(
        min_sigma=1,
        max_sigma=10,
        num_sigma=30,
        threshold=dots_threshold,
        measurement_type='mean',
    )

    # blobs = dots; define the spots in the dots image, but then find them again in the stack.
    intensities = p.run(filtered_imgs, blobs_image=dots, blobs_axes=(Axes.ROUND, Axes.ZPLANE))

    # decode
    decoded = experiment.codebook.decode_per_round_max(intensities)

    # save results
    df = decoded.to_decoded_spots()
    df.save_csv("decoded.csv")


if __name__ == "__main__":
    test_experiment()