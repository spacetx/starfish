## Usage

install miniwdl

.. code-block:: bash

    pip3 install miniwdl


check the workflow

.. code-block:: bash

    miniwdl check workflows/sequential_smFISH/ssmfish.wdl

run the workflow locally on two fields of view to test

.. code-block:: bash

    URL=https://d2nhj9g34unfro.cloudfront.net/browse/20190111/allen_mouse_panel_1/experiment.json
    miniwdl cromwell workflows/sequential_smFISH/ssmfish.wdl \
        num_fovs=2 \
        experiment="${URL}"
