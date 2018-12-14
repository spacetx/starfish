Debugging Errors
================

First, thank you for using Starfish and SpaceTx-Format! Feedback you provide on features and the
user experience is critical to making Starfish a successful tool. Because we iterate quickly on this
feedback to add new features, things change often, which can result in your code getting out of sync
with your data. When that happens, you may observe errors.

Most of the time, you can fix this problem by pulling the most recent version of the code,
reinstalling starfish, and restarting your environment. If you're using starfish with datasets from
spaceTx located on our cloudfront distribution, we're committed to keeping that data up to date.
Updated versions of the notebook will reference the correct data version, and copying over the
new link should fix any issues.

For example, if a notebook references in-situ sequencing data from August 23rd, and a breaking
change occurs on September 26th, it would be necessary to replace the experiment link to point at
data that was updated to work post-update:

.. code-block:: diff

    - http://spacetx.starfish.data.public.s3.amazonaws.com/browse/formatted/20180823/iss_breast/experiment.json
    + http://spacetx.starfish.data.public.s3.amazonaws.com/browse/formatted/20180926/iss_breast/experiment.json

If you're using your own data with starfish, you may need to re-run your data ingestion workflow
based on :py:class:`starfish.experiment.builder.providers.TileFetcher` and
:py:class:`starfish.experiment.builder.providers.FetchedTile` to generate up-to-date versions of spaceTx-format.

Upgrading to a new version
--------------------------

If you've installed from pypi, upgrading is as simple as reinstalling starfish.

.. code-block:: bash

    pip install --upgrade starfish

If you've installed our development version to take advantage of new features in real time, you'll
need to fetch changes and reinstall. Assuming you've cloned the respository into ``./starfish``,
you can install the newest version as follows:

.. code-block:: bash

    cd ./starfish
    git checkout master
    git pull
    pip3 install .

Reporting bugs
--------------

Bugs can be contributed as issues in the starfish repository. Please check to make sure there
is no existing issue that describes the problem you have identified before adding your bug.

When reporting issues please include as much detail as possible about your operating system,
starfish version, slicedimage version, and python version. Much of this can be accomplished by
sending us the output of ``pip freeze``:

.. code-block:: bash

    pip freeze > environment.txt

Whenever possible, please also include a brief, self-contained code example that demonstrates the
problem, including a full traceback.

We can also be contacted on the SpaceTx slack in the ``#starfish-users`` channel. If you'd like an
invitation, please feel free to email us. We can usually respond to bug reports same-day, and
are very appreciative of the time you take to submit them to us.
