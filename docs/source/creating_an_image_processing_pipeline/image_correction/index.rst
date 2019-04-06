.. _image_correction:

Illumination Correction
=======================

The goal of illumination correction is to remove uneven illumination of the image caused by non
uniform illumination of the field of view, characteristics of the sensor, (like vignetting), or
orientation of the tissue's surface with respect to the light source.

The simplest forms of illumination correction are called "prospective correction" and are based on
background subtraction. This involves taking additional images using the microscopy apparatus to
help calibrate. These can either be acquired by averaging a series of images captured with no
sample and no light (dark image), or with no sample and light (bright image).

Starfish can apply this type of background correction by exposing the :py:class:`ElementWiseMult`
:ref:`Filter <filtering>`. The user is responsible for transforming their calibration images into
the correct matrix to correct for background, and then :py:class:`ElementWiseMult` can apply a
transformation to correct any uneven illumination.

# TODO photographic examples and further reading links

Chromatic Aberration
====================

Chromatic Aberration refers to the failure of a lens to focus all colors to the same positions.
Because multiplex spot counting experiments tend to leverage fluorescence signals from different
spectral bands and restrict each color channel to its own image, the resulting problems can be
complex to detect.

Starfish currently exposes some basic registration effects, and these can be enough to correct for
very minor chromatic aberrations. Additionally, non-multiplex approaches may only require that spots
find their way into the correct cell, providing some flexibility to ignore minor aberrations.

However, most multiplex experiments will require some kind of correction beyond what is provided by
starfish's basic translation. At this point in time starfish **does not** provide tooling for the
correction of chromatic aberrations, and as such users must correct for these types of errors
in their data prior to submitting it to starfish. However, we would be very excited to receive
code contributions of filters that solve these problems
(see :ref:`contributing to starfish <contributing>`)

To read more about types of chromatic aberration that can appear in microscopy data, see
`wikipedia`_

.. _wikipedia: https://en.wikipedia.org/wiki/Chromatic_aberration

Deconvolution of Optical Point Spread Functions
===============================================

Deconvolution is a technique that enables a user to reverse any optical distortion introduced by
the microscope. Deconvolution is accomplished by assuming that the path of light through the
instrument is perfect, but convolved with a "point spread function". By deconvolving the image and
the point spread function, the distortion can be removed.

The point spread function can be determined in several ways. Ideally, it is approximated during
calibration of the microscope, in which case it can be removed by the
`Richardson-Lucy algorithm <richardson-lucy-web>`_ (API:
:ref:`Richardson-Lucy <richardson-lucy>`)

.. _richardson-lucy-web: https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution

Learning the PSF from image data
--------------------------------

The simplest way to learn the PSF from spot-based data is to learn the shape of the spots. If we
assume that the spots are derived from point sources of light, the average spot shape can form the
basis for estimating the PSF needed by the richardson-lucy (see above). This depends on the data
having a relatively uniform spot shape, which not all image-based transcriptomics and relatively few
image-based proteomics experiments adhere to.

# TODO incorporate `Nick's vignette <nsofroniew-vignette>`_.

.. _nsofroniew-vignette: https://gist.github.com/sofroniewn/8acccc0d040fc8d9325267c83c8febc9

It can also be estimated from the frequency properties of the experimental data, called
"blind deconvolution", for which variants of both `Richardson-Lucy <richardson-lucy-blind>`_
and the `Weiner filter <wiener>`_ have been proposed. Starfish does not provide any tooling
for blind deconvolution.

.. _richardson-lucy-blind: 10.1364/JOSAA.12.000058
.. _wiener: https://www.sciencedirect.com/science/article/pii/S0377042717306544

To read more about image deconvolution, see this article:
`<https://en.wikipedia.org/wiki/Deconvolution#Optics_and_other_imaging>`

Starfish

Image Registration
==================

Registration is an important aspect of image correction, particularly for multiplex experiments
which attempt to match dots across images and channels. In these experiments, even very small shifts
the size of a dot can make it extremely challenging to identify gene that dot present across a set
of images.

As mentioned above, chromatic aberration is one form of error that spot-calling experiments must
contend with. However, registration error can also occur from tissue handling during the experiment,
subtle changes in the position of the microscope slide relative to the stage, microfluidics shifting
the tissue during fluid exchange across rounds, or changes in tissue morphology in expansion
microscopy experiments.

Starfish exposes simple fourier-domain translational registration to adjust for *some* of the above
registration issues. Starfish also supports the Warp functionality to apply any
pre-learned affine transformation, The combination of these approaches covers all of the types of
registration faults that starfish's developers have observed thus far while looking at image-based
transcriptomics experiments. However, it will be necessary to learn registrations that are more
complex than translational shifts using software outside starfish, for the time being. Any
contributions that add registration features would be highly desirable.

# TODO ambrosejcarr examples and links to warp

Example Image Correction Pipeline
=================================

# TODO put together a worked example in the gallery and link it here.
