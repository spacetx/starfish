#!/usr/bin/env python

import os
import setuptools
import versioneer

install_requires = [
    line.rstrip() for line in open(os.path.join(os.path.dirname(__file__), "REQUIREMENTS.txt"))
]

setuptools.setup(
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    extras_require={
        'napari': ['napari==0.0.6'],
    },
    entry_points={
        'console_scripts': [
            "starfish=starfish:starfish",
        ]
    },
    include_package_data=True,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
