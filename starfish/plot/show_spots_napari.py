from typing import Mapping, Union

from numpy import array, zeros

from starfish import IntensityTable
from starfish.types import Indices


def show_spots_napari(
	spots: IntensityTable=None,
	background_image=None,
	radius_multiplier=30):

	try:
		import napari_gui
	except ImportError:
		warnings.warn("Cannot find the napari library. "
						"Install it by running \"pip install napari-gui\"")
		return


	c_r = list(zeros(len(spots.y.values)))
	coords = array([spots.x.values, spots.y.values, c_r, c_r, spots.z.values]).T

	mp = background_image.max_proj(Indices.CH, Indices.ROUND)

	viewer, axes = mp.show_stack_napari({})

	sizes = spots.radius.values * radius_multiplier

	# Add the markers to the viewer
	viewer.add_markers(coords, face_color='white', edge_color='white', symbol='ring', size=sizes)

	return viewer