from starfish.spots import SpotFinder
from starfish.image import Filter
from starfish import Experiment, Codebook


kwargs = dict(
    spot_diameter=5,  # must be odd integer
    min_mass=0.02,
    max_size=2,  # this is max _radius_
    separation=7,
    noise_size=0.65,  # this is not used because preprocess is False
    preprocess=False,
    percentile=10,  # this is irrelevant when min_mass, spot_diameter, and max_size are set properly
    verbose=True,
    is_volume=True,
)
tlmpf = SpotFinder.TrackpyLocalMaxPeakFinder(**kwargs)
bandpass = Filter.Bandpass(lshort=.5, llong=7, threshold=0.0)
sigma = (1, 0, 0)  # filter only in z, do nothing in x, y
glp = Filter.GaussianLowPass(sigma=sigma, is_volume=True)
clip1 = Filter.Clip(p_min=50, p_max=100)
clip2 = Filter.Clip(p_min=99, p_max=100, is_volume=True)


exp = Experiment.from_json('/Users/shannonaxelrod/Desktop/brian/experiment.json')
codebook = Codebook.from_json('/Users/shannonaxelrod/Desktop/brian/codebook.json')
primary_image = exp.fov()['primary']
new_image = clip1.run(primary_image, verbose=True, in_place=False, n_processes = 7)
bandpass.run(new_image, verbose=True, in_place=True, n_processes = 7)
glp.run(new_image, in_place=True, verbose=True, n_processes = 7)
clip2.run(new_image, verbose=True, in_place=True, n_processes = 7)
spot_attributes = tlmpf.run(new_image)
decoded = codebook.decode_per_round_max(spot_attributes)
decoded[decoded["total_intensity"]>.025]