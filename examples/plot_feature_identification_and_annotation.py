"""
Feature Identification
======================

"""

####################################################################################################
# .. _tutorial_segmenting_cells:
#
# Segmenting Cells
# ================
#
# Cell segmentation is a very challenging task for image-based transcriptomics experiments. There
# are many approaches that do a very good job of segmenting *nuclei*, but few if any automated
# approaches to segment *cells*. Starfish exposes the watershed segmentation method from classical
# image processing, which inverts the intensities of the nuclei and spots and treats them like a
# literal watershed basin. The method then sets a threshold (water line), and each basin is treated
# as its own separate segment (cell).
#
# This approach works fairly well for cultured cells and relatively sparse tissues, but often cannot
# segment denser epithelia. As such, starfish *also* defines a simple segmentation format to enable it
# to read and utilize segmentation results derived from hand-drawing or semi-supervised drawing
# applications.
#
# TODO cell segmentation demo
#
pass

####################################################################################################
# .. _tutorial_assigning_spots_to_cells:
#
# Assigning Spots to Cells
# ========================
#
# TODO cell assignment demo and creating Cell x Gene matrix

pass
