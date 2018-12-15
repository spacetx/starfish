from numpy import zeros, array

from starfish import IntensityTable
from starfish.types import Indices


def show_spots_napari(
	spots: IntensityTable=None,
	background_image=None,
	radius_multiplier=30):

	mp = background_image.max_proj(Indices.CH, Indices.ROUND)

	# Make the array of maker coordinates
	# If the background image is z projected, also z project the coordinates
	c_r = zeros(len(spots.y.values))
	if mp.raw_shape[2] == 1:
		coords = array([spots.x.values, spots.y.values, c_r, c_r, c_r]).T

	else:
		coords = array([spots.x.values, spots.y.values, c_r, c_r, spots.z.values]).T
		
	# Create the Napari viewer with an image stack
	viewer, axes = mp.show_stack_napari({})

	# This initializes an index for the instantiated marker display
	# Should be fixed in Napari in the future - KY
	viewer._index = [0,0,0,0,0]

	# Get the sizes
	sizes = spots.radius.values * radius_multiplier

	# Add the markers to the viewer
	viewer.add_markers(
						coords=coords, face_color='white', edge_color='white',
						symbol='ring', size=sizes)

	return viewer
