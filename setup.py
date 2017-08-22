import setuptools

setuptools.setup(
    name="starfish",
    version="0.0.1",
    description="prototype a standardized pipeline for the analysis of image-based transcriptomics data",
    author="Deep Ganguli",
    author_email="dganguli@chanzuckerberg.com",
    license="MIT",
    packages=["starfish"],
    entry_points={
        'console_scripts': "starfish=starfish.starfish:starfish"
    }
)
