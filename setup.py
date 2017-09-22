import setuptools

setuptools.setup(
    name="starfish",
    version="0.0.1",
    description="Pipelines and pipeline components for the analysis of image-based transcriptomics data",
    author="Deep Ganguli",
    author_email="dganguli@chanzuckerberg.com",
    license="MIT",
    packages=setuptools.find_packages(),
    entry_points={
        'console_scripts': "starfish=starfish.starfish:starfish"
    }
)
