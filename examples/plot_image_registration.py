"""
.. _tutorial_image_registration:

Image Registration
==================
In order to target more genes than the number of spectrally distinct fluorophores, assays use
multiple rounds of imaging. But experimental conditions between imaging rounds such as microscope
stage drift, movement of the microscope slide on the stage, microfluidics shifting the tissue during
fluid exchange, and changes in tissue morphology cause the field of view to shift.

Correcting this error by registering images to a common reference image is important for accurate
localization of RNA targets and partitioning spots into single cells. It is especially critical for
exponentially multiplexed assays that trace spots across rounds to decode a barcode or sequence,
since it matches spots by their positions in every round.

There are a few options to address the shifting between image rounds:

1. **Ignore it.** If the shift is minor (i.e. the largest shift is much less than the diameter of a
cell) and your assay does not require spots to be aligned across rounds to decode
barcodes or sequences, then you can skip image registration. Be aware that this option limits you to
using :py:class:`.SimpleLookupDecoder` when decoding.

2. **Register your images outside of starfish.** If you need more complex transformations than
starfish has available, you can register your images prior to loading them into the starfish
pipeline.

3. **Use starfish.** Starfish exposes simple fourier-domain translational registration to adjust
some common registration issues. It also supports the :py:class:`.Warp` functionality to apply any
pre-learned affine transformation.

This tutorial will cover how to use :py:class:`.LearnTransform` and :py:class:`.ApplyTransform` to
register `primary images` in a starfish pipeline. This in situ sequencing (ISS) example includes a
`dots` image, which is one image with all the RNA spots, that is used as the `reference_image` for
registration. The maximum intensity projection of any round from `primary images` can also be
used in lieu of a `dots` image.
"""

from starfish import data

experiment = data.ISS(use_test_data=True)
imgs = experiment["fov_001"].get_image('primary')
dots = experiment["fov_001"].get_image('dots')

###################################################################################################
# The images used by :py:class:`.LearnTransform` depends on the data available. For example,
# if there are common landmarks present in every `primary image`, such as fiducial markers (e.g.
# fixed fluorescence beads) or autofluorescence cellular structures, then those images can be used
# to learn the transforms. In this example where all RNA spots are present in each round,
# the images from each round are projected and then the RNA spots are used as the landmarks.

from starfish.types import Axes

projected_imgs = imgs.reduce({Axes.CH}, func="max")
print(projected_imgs)

###################################################################################################
# This ISS example has 4 rounds of `primary images` that need to be registered. Starfish provides
# a utility for plotting each round with a different color overlaid on the same axes. Here we can
# see that there is a uniform shift between rounds at all regions of the FOV, which suggests a
# translational transformation should be able to register the images.

import matplotlib
from starfish.util.plot import diagnose_registration

matplotlib.rcParams["figure.dpi"] = 250
diagnose_registration(projected_imgs, {Axes.ROUND:0}, {Axes.ROUND:1}, {Axes.ROUND:2}, {Axes.ROUND:3})

###################################################################################################
# Next we learn the translational transform using :py:class:`.LearnTransform.Translation`,
# which wraps :py:class:`skimage.feature.register_translation` for efficient image translation
# registration by cross-correlation. Running it will find the translational shift of every image
# along a specified :py:class:`.Axes` of an :py:class:`.ImageStack` relative to the
# `reference_image`.
#
# .. Note::
#   There must be more landmarks than random noise. Otherwise the
#   :py:class:`.LearnTransform.Translation` could align random noise.

from starfish.image import LearnTransform

learn_translation = LearnTransform.Translation(reference_stack=dots, axes=Axes.ROUND, upsampling=1000)
transforms_list = learn_translation.run(projected_imgs)

###################################################################################################
# We can save the `transforms list` as a json file for future use or view it to investigate the
# registration.

transforms_list.to_json('transforms_list.json')
print(transforms_list)

###################################################################################################
# Currently, starfish can only learn translational transforms but can *apply* any
# `geometric transformations`_ supported by :py:class:`skimage.transform.warp`. If you need more
# complex registrations, you can load them as a :py:class:`.TransformsList` from a json file.
#
# .. _geometric transformations: https://scikit-image.org/docs/dev/auto_examples/transform/plot_geometric.html#sphx-glr-auto-examples-transform-plot-geometric-py_
#
# .. code-block::
#
#   from starfish.core.image._registration.transforms_list import TransformsList
#
#   transforms_list = TransformsList()
#   transforms_list.from_json('transforms_list.json')

###################################################################################################
# Applying the transform to an :py:class:`.ImageStack` is done with
# :py:class:`.ApplyTransform.Warp`, which wraps :py:class:`skimage.transform.warp`.
# Note that the transform was learned from the *projection* of the `primary images` but
# is being applied to the *original* `primary images` :py:class:`.ImageStack`. The transform for
# each round will be applied to every channel and zslice.

from starfish.image import ApplyTransform

warp = ApplyTransform.Warp()
registered_imgs = warp.run(imgs, transforms_list=transforms_list, in_place=False)

###################################################################################################
# Validate that the registration was successful.

diagnose_registration(registered_imgs.reduce({Axes.CH}, func="max"), {Axes.ROUND:0}, {Axes.ROUND:1}, {Axes.ROUND:2}, {Axes.ROUND:3})

###################################################################################################
# Additional considerations for image registration:
#
# Although in this example, we both learned and applied the transform to the `primary images`,
# this is not the only way. Some assays don't have any reliable landmarks present in every round of
# the `primary images`. In that case, another option is to acquire an additional channel with a
# counterstain (e.g. DAPI) or brightfield image for every round. The transform is learned from
# those images and then applied to the `primary images`. The one caveat to using this method is
# that the RNA spots must be reliably fixed in place relative to images used for learning the
# transform.
#
# After registering the images, the dimensions of the :py:class:`.ImageStack` is the same and the
# physical coordinates are that of the `reference_image`. Regions of images from each round that
# aren't overlapping are either cropped out or filled with zero values. To keep only the
# overlapping region of the :py:class:`.ImageStack` you can :ref:`crop <tutorial_cropping>`,
# which will crop the physical coordinates as well.
