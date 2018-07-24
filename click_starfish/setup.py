#!/usr/bin/env python

import setuptools


setuptools.setup(
    name="click_starfish",
    version="0.0.0",
    description="Proof of concept for starfish using the click cli library",
    author="Ambrose J. Carr",
    author_email="mail@ambrosejcarr.com",
    license="MIT",
    entry_points={
        'console_scripts': "click-starfish=click_starfish.cli:cli"
    },
    install_requires=['click']
)
