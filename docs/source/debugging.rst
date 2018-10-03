Debugging Errors
================

First, thank you for using Starfish and SpaceTx-Format! Feedback you provide on features and the
user experience is critical to making Starfish a successful tool. Because we iterate quickly on this
feedback to add new features, things change often, which can result in your code getting out of sync
with your data. When that happens, you may observe errors.

Most of the time, you can fix this problem by pulling the most recent version of the code,
reinstalling starfish, and restarting your environment. If you're using starfish with datasets from
spaceTx located on our cloudfront distribution, we're committed to keeping that data up to date.
You can find it in a versioned folder, and the version will correspond to the version of starfish
that it matches.

If you're using your own data with starfish, you may need to re-run your data ingestion workflow
based on TileFetcher and FetchedTile to generate up-to-date versions of spaceTx-format.

Upgrading to a new version
--------------------------

If you've installed from pypi, upgrading is as simple as reinstalling starfish.

.. code-block:: bash

    pip3 install --upgrade starfish

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
starfish version, slicedimage version, and python version. Whenever possible, please also include a
brief, self-contained code example that demonstrates the problem, including a full traceback.

We can also be contacted on the SpaceTx slack in the ``#starfish-users`` channel. If you'd like an
invitation, please feel free to email us. We can usually respond to bug reports same-day, and
are very appreciative of the time you take to submit them to us.
