.. _data structures:

Data Structures
===============

The top-level object in a starfish workflow is the :ref:`Experiment`. It is composed of one or more
:ref:`FieldOfView` objects, and a :ref:`Codebook`, which maps detected spots to the entities they
target.

Each :ref:`FieldOfView` consists of a set of Primary Images and optionally, Auxiliary images that
may contain information on nuclei (often used to seed segmentation) or fiduciary beads (often used
to enable fine registration).

Both Primary and Auxiliary Images are referenced by slicedimage_ TileSet_ objects, which map
two dimensional image tiles stored on disk into a 5-dimensional Image Tensor that labels each
``(z, y, x)`` tile with the ``round`` and ``channel`` that it corresponds to. When loaded into
memory, these Image Tensors are stored in :ref:`ImageStack` objects. The :ref:`ImageStack` is what
starfish uses to execute image pre-processing, and serves as the substrate for spot finding.

Identified spots are stored in the :ref:`IntensityTable`, which stores the intensity of the spot
across each of the rounds and channels that it is detected in. It also stores assigned genes when
decoded with a :ref:`Codebook` and assigned cells when combined with a segmentation results.

Finally, the :ref:`IntensityTable` can be converted into an :ref:`ExpressionMatrix` by summing all
of the spots detected for each gene across each cell. The ExpressionMatrix provides conversion and
serialization for use in single-cell analysis environments such as Seurat_ and Scanpy_.

.. TODO ambrosejcarr: think about removing PipelineComponent to another part of the docs

.. _slicedimage: https://github.com/spacetx/slicedimage

.. _TileSet: https://github.com/spacetx/slicedimage/blob/master/slicedimage/_tileset.py

.. _Seurat: https://satijalab.org/seurat/

.. _Scanpy: https://scanpy.readthedocs.io/en/latest/

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. toctree::
   experiment.rst

.. toctree::
   field_of_view.rst

.. toctree::
   image_stack.rst

.. toctree::
   codebook.rst

.. toctree::
   expression_matrix.rst

.. toctree::
   intensity_table.rst

.. toctree::
   decoded_intensity_table.rst

.. toctree::
   binary_mask.rst

.. toctree::
   label_image.rst
