# _starfish_ 0.1.0 Roadmap
This document describes the features required for SpaceTx groups to use _starfish_ to analyze their data. These users will generate feedback on _starfish_, enabling work to be tailored towards the most valuable feature set.

The purpose of this document is to arrive at a detailed list of short-term key deliverables that we aim to deliver by July 2019, the targeted release date for 0.1.0. To accomplish this, we first outline our primary use cases. With these to guide us, we then outline [**what we need to build**](#what-we-need-to-build:-starfish-0.1.0), [**what we can build later**](candidates-features-for-0.2.0+), and why.

## Primary use cases
To support SpaceTx and Chan Zuckerberg Biohub benchmarking efforts, _starfish_ needs to support the following use cases.

### Parameter selection on single fields of view
Image-based transcriptomics workflows are highly dependent upon the tissue, organism, and probes being assayed. Labs typically select image processing parameters by analyzing and optimizing parameter selection in a single field of view. In support of this use case, _starfish_ must enable users to process data for any SpaceTx assay on the local computer of their choosing and interact with visualization tools necessary to evaluate parameter decisions. It must support this functionality for users that are not proficient in python.

### Local processing of data for SpaceTx Users
Some SpaceTx Assays have moderate processing requirements. Users of these assays would like to be able to use _starfish_ to process multiple fields of view of SpaceTx data on their local machines to enable further parameter tuning and spot-checking of results. To support this, we will define a Python API for running _starfish_ on local machines that processes experiments of arbitrary numbers of fields of view, logs, stores data provenance, is multi-processing enabled. Users will also be able to run _starfish_ pipelines through a command-line interface (CLI).

### Scalable processing of SpaceTx assays
For the SpaceTx consortium project, the _starfish_ team agreed to process the data for each of the data contributors. It may not always be possible or time efficient to do so on a local machine. To accomplish this, _starfish_ must be able to process a complete experiment consisting of hundreds of fields of view that are comprised of up to 10s of terabytes of 2-d TIFF images ("Scalable Pipeline Runner"). We will run these analyses, so can dictate the infrastructure on which this processing will occur. However, we need to expose tooling to explore the output formats on a personal computer of our contributor's choosing. The Chan Zuckerberg Biohub workflow is maintained by a single local user, for whom we can help set up this workflow runner.

### Scientific assessment of analyzed SpaceTx data: cell type mapping
The SpaceTx consortium Working Group 5 is devoted to mapping the cells in Cell x Gene tables into the transcriptomic types/clusters delivered by Working Group 2.  _starfish_ will define standard output formats for external (non-_starfish_) calculation of these cell types and subsequent incorporation of cell type information as metadata in the Cell x Gene tables.

### Scientific assessment of analyzed SpaceTx data: methods comparison and meta analysis
The SpaceTx consortium Working Group 6 is devoted to comparing the results of each method and assessing their relative strengths and weaknesses. These are largely scientific analyses enabled by the standardization of each assay's outputs by _starfish_.  Because each assay outputs an IntensityTable, Segmentation Mask, and Expression Matrix, it will be simple to label cells or spots with anatomical information represented as named 2D polygons.

This capability will allow straightforward creation of highly compressed datasets that will enable working Group 6 to ask questions across all methods like "how many cells/mm2 belonging to the Sst-Chodl cluster were found in layer 4 of mouse primary visual cortex?"  or "Within the inhibitory cell class, what is the abundance and distribution of Pvalb+ and VIP+ cells in human cortex?"

Finally, with a standard format to save and load Cell x Gene tables, visualization of this data will be relatively straightforward for analysts in contributor labs and Working Group 6 members.

## Key Deliverables
We believe that delivering the following features will enable the above use cases.

1. Data is easily converted into SpaceTx format from common imaging formats
2. Library that can be used to reproduce collaborator's results for single fields of view
3. Library is usable by SpaceTx for generating pipeline recipes on a single or small number of FOVs (minimum) or processing an entire experiment on individual machines (at best)
4. A way for the _starfish_ team to run _starfish_ at scale to process the initial set of SpaceTx datasets
5. Documentation that supports new users to learn to use _starfish_ without our involvement
6. _starfish_ facilitates easy interaction with external visualization software (Napari, FIJI) to enable users to tune image processing parameters
7. Experiment-scale output formats can be loaded on local computers and are adequately expressive to enable QC workflows and specified by our users.
8. _starfish_ output formats enable external annotation to place the data in anatomical and cell type context and allows comparison across methods

Each deliverable above can be broken up into a set of milestones
### Library that can be used to reproduce collaborator's results for single fields of view
Where parenthetical fractions are listed, they represent progress towards our goals at the time this roadmap was written.

1. Obtain example data, pipelines, and results from each SpaceTx group (9/10)<sup>[1](#fn1)</sup>
2. Create a data conversion tool to wrangle contributed data into SpaceTx format (0/10)
3. Specify an input file format and corresponding object model that supports each assay type (8/10)
4. Specify output file format specifications for detected spots, gene expression matrices, and detected objects (e.g. cells) (0/3), and corresponding object models  (2/3)<sup>[2](#fn2)</sup>
5. Implementation, in python, of a single-fov proof of concept pipeline that closely matches the processing results of data generators for 9 SpaceTx groups. This is computational biology work to understand the characteristics of the data & assay, and can be done outside _starfish_ as necessary to identify its requirements. (3/10)<sup>[3](#fn3)</sup>
6. Implementation, in _starfish_, of the same single-FOV pipelines above (3/10)

### Library is usable by SpaceTx for generating pipeline recipes (minimum) or processing their data on individual machines (at best)
1. Library is easily installed on operating systems used by SpaceTx users
2. Basic for-loop implementation of a pipeline runner to de-risk scale problems that result from processing multiple fields of view, then integrating results for biological analysis
3. Explicit definition/specification of a pipeline recipe
4. API for processing multiple FOVs for each SpaceTx group given data in SpaceTx format and a pipeline recipe
5. Translate each example pipeline into an example pipeline recipe
6. Visualization tooling to enable parameter selection on small datasets to tweak pipeline recipes
7. API leverages local parallelism

### A way to run _starfish_ at scale to process SpaceTx datasets
1. _starfish_-based solution to process multiple FOVs at scale tied to a specific infrastructure (Not designed to be run by SpaceTx users on their hardware) (0/9)<sup>[4](#fn4)</sup>

### Documentation such that new users can learn to use it without our involvement
1. Public and Private API is fully typed and documented
2. Purpose of each pipeline component is described, and guidance on parameter tuning, if necessary, is provided
3. There are examples of formatting data in SpaceTx format
4. There are examples of fully-worked pipelines
5. There are examples of the workflow for creating new pipelines
7. Public API is stable

### _starfish_ facilitates easy interaction with external visualization software (Napari, FIJI) to enable users to tune image processing parameters
1. _starfish_ has a method to visualize an ImageStack in Napari
2. _starfish_ has a method to plot spots from an IntensityTable onto an ImageStack in Napari
3. _starfish_ can dump ImageStacks for viewing 2D TIFFs in FIJI

### Experiment-scale output formats can be loaded on local computers and are adequately expressive to enable QC workflows specified by our users. These output formats enable scientific investigation of the data.
1. _starfish_ is used to process the three example datasets that we have access to:
    1. ISS Breast cancer. 16 FOVs with shape (4, 4, 1, 1024, 1024),
    2. MERFISH U2-OS cells. 400 FOVs with shape (8, 2, 1, 2048, 2048),
    3. osmFISH Visual Cortex. ~100s FOVs with shape (13, 3, 45, 2048, 2048)
2. Outputs from Fields of view can be combined into a single output format
3. Output formats can be loaded and are performant
4. Processed data can be used to recapitulate results

### _starfish_ output formats enable external annotation to place the data in anatomical and cell type context and allow comparison of across methods
1. Processed Experiments (all FOVs for a given dataset) can produce Cell x Gene tables with each cell in the physical coordinates of the sample
2. 2D polygon annotations of anatomical structures in physical coordinates can be saved in a standardized format and membership of each cell in annotation structures can be added as metadata to the Cell x Gene table
3. Cell x Gene table can be easily imported by computational biologists (outside of _starfish_) to determine cell type or cluster membership for each cell.  This cluster membership can include probabilistic membership to multiple clusters and is added to the Cell x Gene table as metadata

### A tool to convert image data into SpaceTx Format
1. CLI and API (Java) can parse [supported file formats](https://docs.openmicroscopy.org/bio-formats/6.0.0-m3/supported-formats.html) into SpaceTx-formatted datasets<sup>[5](#fn5)</sup>
2. Installation of the tool can be accomplished by downloading an archive with a launcher script or by using pre-built docker images
3. Users and/or data wranglers can point the tool at one of the [stated files](https://docs.openmicroscopy.org/bio-formats/6.0.0-m3/formats/dataset-table.html) for each FOV to generate 2D TIFFs as well as the necessary SpaceTX JSON files
4. In the case that multiple “series” are present in a single dataset, the user must specify which series to use
5. The index of the field of view (0-based) must be passed during creation
6. If a codebook exists at the time of creation, it can be passed to the tool for inclusion in the JSON. Otherwise, a dummy codebook will be created

## What we need to build: _starfish_ 0.1.0
In general, we aim for _starfish_ to be a comprehensive platform for the processing of image-based transcriptomics experiments. It should be intuitive to use, fast, and should not require extensive experience with python programming. However, at least during the initial development period, _starfish_ will require domain experience in image processing, as our target users are experts in image generation and processing.

### A General Format and Object Model for Image-based Transcriptomics
_starfish_ defines a general set of data formats (and paired python object implementations) that (1) enable efficient storage and access for a related set of images broken up into single fields of view<sup>[6](#fn6)</sup>, (2) enable storage and utilization of a codebook, which defines how sets of images are combined to decode spot patterns into biological targets, and (3) defines a pipeline recipe which can be interpreted by _starfish_'s pipeline runner to process an experiment.

To be eligible for processing, images in SpaceTx Format must either be pre-aligned, such that the x-y location of each tile in a Field of View is the same. Image data that does not adhere to this requirement will not be processable with _starfish_ without addition of user-contributed registration pipeline components.

### A Modular Library of Pipeline Components
_starfish_ enables the construction of arbitrary image-based transcriptomics pipelines by implementing the union of algorithms used by SpaceTx labs while combining similar approaches into a simplified subset. These algorithms can be linearly combined to process individual fields of view, requiring no more than 16GB ram for any individual pipeline component. The set of algorithms must include:

#### Image Adjustment
_starfish_ defines pipeline components that enable simple adjustment of image data, such as slicing (finding focal plane, cropping edge effects) and projection over arbitrary axes.

#### Filtering
_starfish_ defines the set of filters used in SpaceTx labs, including approaches to remove background, enhance spots, deconvolve the specified point spread function of the imaging system, and normalize image intensities.

#### Spot Detection
_starfish_ implements spot detectors via a variety of spot detection algorithms (eg., Gaussian/Laplacian of Gaussian/Hessian of Gaussian blob detection, local max peak finding, etc.) These detectors can function independently on individual channels and rounds or be used to measure intensity values of spots (as detected in a single auxiliary image) across all other primary images across channels and rounds. _starfish_ also supports pixel based decoding algorithms (e.g., algorithms that detect spots after pixels are decoded to targets) within the same API.

#### Decoding
_starfish_ enables decoding of spots or pixels detected across rounds and channels into targets. These decoding results can be filtered post-hoc, using spot intensity, spot size, and spot distance-to-target metrics. For the latter metric, the user may supply their desired distance metric.

#### Segmentation
This is an area of active research. However, four SpaceTx groups use seeded watershed segmentation, which has a concrete implementation. _starfish_ implements this method and will include a vignette on how to tune parameters so it functions across assays. In addition, we will elicit submissions from Kenneth Harris and Peter Kharchenko for point-cloud based segmentation, and encourage the community to work on this problem by defining a segmentation format and API to enable plugin-based interaction with community-developed algorithms.

#### Image Registration
_starfish_ implements the ability to apply a pre-computed full affine registration including translation, rotation, and scale to enable alignment across z-layers, rounds, and channels. Translation and rotation can also provide a solution for some types of chromatic aberration.

### Worked examples that process each assay
We need to demonstrate that _starfish_ is capable of processing each SpaceTx assay. To do this, we will create pipelines that process single fields of view for each group. These pipelines will take as input datasets that have already been registered and for which chromatic aberrations have been corrected, and will output spot calls. To validate that our results match SpaceTx expectations, we will calculate the copy number correlation between our results and the results of their own pipelines, as well as overlay our spot calls on the image data.

### Pipeline Recipes
_starfish_ will define a format (or elect an existing one) for a pipeline recipe. Such a format must consist of, at a minimum, a selection and ordering of pipeline components and a specification of any necessary parameters. We will create example recipes for each group, and will solicit recipes from each SpaceTx group that are correctly parameterized to process (1) human brain tissue and (2) mouse brain tissue for each of their respective assays.

### An Example Pipeline Runner
A pipeline runner takes as input a pipeline recipe, runs the pipeline according to this recipe, and outputs combined results (combined across fields of view) in a standardized file format. There are two versions of a runner: one that runs on a single machine, and one that distributes computation across multiple machines.

#### Single-machine Pipeline Runner
An un-optimized executor of pipeline recipes that logs and records the execution of pipeline components across FOVs to create a reproducible record of how data was processed. Ideally this runner would be configurable for multi-processing, when relevant. This runner must enable SpaceTx users to easily tune parameters on single fields of view.
#### Scalable Pipeline Runner
There are many existing distributed pipeline runners that are designed for particular architectures (AWS batch, Google Pipelines API, Grid Engine), and frameworks that build programmatically on top of them (Spark, Dask, Hadoop), or abstract graph computation and scheduling over pools of potentially distributed compute resources (WDL/Cromwell, Toil, Snakemake, Nextflow, CWL, Reflow, ...). There is also an open question as to whether distributed or GPU-based architectures will be more cost effective for image-based transcriptomics processing.

The diversity of infrastructures leveraged by SpaceTx labs and institutes have caused them to make different decisions about which workflow runner to use (EBI: HPC+Nextflow, UCSC: ?+Toil, Broad: GCP+Cromwell, Allen: HCP+Grid Engine, Zhuang: HCP+Grid Engine+SnakeMake, etc.). Because institutional users tend to be heavily committed to their infrastructure of choice, it is unlikely that we will convert them. However, by implementing a library that can be leveraged by each workflow runner, we enable _starfish_ to be used across these groups.

Nevertheless, to demonstrate _starfish_'s scale, we will implement workflows on one or more existing pipeline runner and keep this work distinct from the underlying library. This pipeline runner will be capable of processing a complete experiment consisting of arbitrary numbers of fields of view for any SpaceTx lab and will leverage cloud-based parallelism to accomplish the processing of any SpaceTx experiment in under 2 hours. We will consider the HCA use case when designing our version 0.1.0 workflows in order to maximize initial compatibility.

### A Developer Focus Point
Two key aims of _starfish_ and SpaceTx are to (1) encourage convergence of the community onto a single object model for processing of image-based transcriptomics data and (2) make it easier to develop pipelines by providing a high-performance platform. To facilitate this, we must encourage developers to work on _starfish_ through active outreach and by being encouraging and helpful.

## Outstanding work to complete 0.1.0

### A General Format and Object Model for Image-based Transcriptomics

#### Format specification for IntensityTable and ExpressionMatrix:
These two objects currently lack format specifications, which has confused some users, including Ken Harris. A specification for these formats will help computational users to understand how to consume our data formats, and will help facilitate their conversion for use in other languages. For users who are comfortable with python but not necessarily with xarray, we will build in a series of `to_array` and `to_pandas` options.

#### _starfish_ must enable processing of fields of view by round
Sequential smFISH assays that capture volumetric images produce fields of view that are too large for personal computers<sup>[7](#fn7)</sup>. However, if we are able to break up the processing of this data by round, we could hit our scale targets. Second, multiplexed assays (e.g. MERFISH, SeqFISH, ISS) require all rounds and channels to be loaded into memory to decode. In both cases, we should derisk decoding of these assays by understanding the maximum ImageStack size we can support before we need to implement solutions to manage memory consumption.

### A Modular Library of Pipeline Components

#### Registration
We lack solutions for registration that fulfill requirements for ISS (the Harris version), smFISH, MERFISH. Additionally, expansion microscopy requires non-affine thin-plate spline registration. Registration is out of scope for the timeline expected by SpaceTx to have all the data processed. As such, we expect all contributors to provide us with either pre-registered data, or the registration pipeline component implementations they require. We will need to develop solutions or compromises for the MExFISH and ISS (Harris) methods.

#### Documentation
_starfish_'s API documentation is minimal. We need to add basic documentation that describes how to use each component, and how to fit parameters. This should be adequate for a skilled computational user (Brian Long, Kenneth Harris) to pick up _starfish_ and use it to make a pipeline. _starfish_ should also clearly document how a developer can contribute code they need for their analyses to the project. We will need to source feedback from users on what parts of the documentation provoke confusion.

#### Image re-scaling & normalization
Several approaches normalize images to overcome different channel intensities or round biases. We should implement a pipeline component to support this that samples from images within or across fields of view. Determine the minimum necessary sampling level to achieve needed normalization.

#### Stitching
Decide on and Implement a simple solution for stitching IntensityTables created from tiles that overlap in physical space. This could be as simple as "clobber-left".

#### Workflow logging
Implement tracking of pipeline component execution on ImageStacks that record the analysis that has been done on an experiment to reach its current state.

#### Decoding
SeqFISH implements a decoder that carries out a local search of adjacent pixels to allow for jitter in their decoding. We will need to implement this method or demonstrate that it is unnecessary by leveraging an existing decoder to reproduce their results.

### Worked examples that process each assay
- [ ] Design an example pipeline for BaristaSeq
- [ ] Design an example pipeline for SeqFISH
- [ ] Design an example pipeline for osmFISH
- [ ] Design an example pipeline for StarMAP
- [ ] Work with Allen Institute to ingest their data and run their pipeline for 3D smFISH
- [ ] Source recipes from each SpaceTx contributor.

### A Single-machine Pipeline Runner
- [ ] Given a pipeline recipe, executes a series of _starfish_ PipelineComponents to process a single field of view.
- [ ] Simplify the creation of new pipelines by creating an interpreter that takes a pipeline recipe and generates any necessary intermediate artifacts needed to execute the pipeline on our chosen runner.
- [ ] Extend the pipeline runner to process and integrate multiple fields of view into a single coherent output that enables scientific analyses of the outputs.
- [ ] Must be easy to use and serve users that are not confident python programmers.
- [ ] Should support parallelism.

### Scalable Pipeline Runner
- [ ] Research existing solutions and decide whether we must invest in a distributed pipeline runner to hit our initial speed/cost goals.
- [ ] Implement a pipeline runner that achieves the speed goals for each assay, enabling the _starfish_ team to process all the data from the SpaceTx contributors.

### Pipeline Recipes
- [ ] Research the space of existing pipeline specifications
- [ ] Determine an implementation that, given a selection and ordering of pipeline components and a specification of any necessary parameters, provides sufficient instructions to _starfish_ to process data in SpaceTx Format
- [ ] Define the format for the pipeline recipe and add it to SpaceTx Format
- [ ] Translate the single-fov pipelines into pipeline recipes to serve as examples that SpaceTx groups can build from

### A Developer Focus Point
- [ ] Encourage groups we meet to contribute development resources to _starfish_
- [ ] Support image-based transcriptomics and proteomics developers who are interested in working with us on _starfish_ and have implemented methods that could synergize with _starfish_ by scoping how they could contribute, and supporting any eventual pull requests made to _starfish_.
- [ ] Serve Data To Community
- [ ] CC-by licensing
- [ ] Documented upload procedures to s3

## Candidate features for 0.2.0+

There are many other features that are good candidates for future releases. We discuss these features, and why we believe they can be built after release 0.1.0.

### Hardware- and Acquisition-related corrections
Data generators understand their hardware, chemistry, and samples much better than we do. For now, it's difficult for the _starfish_ team to solve image pre-processing problems that are specific to the signal acquisition challenges each data generator faces. Furthermore, proper image pre-processing is critical to the success of the downstream pipeline components that we're building in the roadmap defined above. Instead of relying on data generators to provide us with image-pre-processing pipeline component implementations, we're requesting that data generators pre-process their images before uploading. This will accelerate progress towards the project's goal of comparing the results from all the methods using standard file formats, and scalable, reproducible image processing workflows. There are three categories of pre-processing that we ask data generators to apply:

#### Affine Registration
For the SpaceTx pilot project, it is adequate to apply pre-computed transformations or leverage already-registered data. Later, when we want to support groups to use _starfish_ for their own research, it will be important to implement affine registration, as otherwise _starfish_ serves only part of their use cases, and they are incentivized to continue running their existing pipelines.

#### Non-Affine Registration
Non-Affine registration approaches are very specific to the types of deformations that the tissue suffers, and have variable forms. We should investigate defining some kind of API for arbitrary transformations that could allow users to add approaches that they require, but we won't do the work to implement these as they don't have broad applicability.

#### Fixing Chromatic Aberration
Chromatic aberrations, including differences in illumination, cross-talk between channels, and channel drift tend to be specific to the assay and microscope, making creation of a general solution complicated. Additionally, these problems are often fixed early in pipelines so it is easy to extricate this step from the pipeline. For these reasons, we will ask the groups to generate data that has already been corrected for release 0.1.0.

### Updating proof-of-concept pipelines with new modules
For example, as work is completed on channel scaling and registration, we would have an opportunity to update the MERFISH pipeline. This work should be done by the users who are requesting the features.

### A Complete Testing Framework
We should continue to design tests as needed, and later we should devote some effort to ensuring that our testing suite is strong enough to catch any errors that are contributed by external developers. However, we can prioritize this after we have succeeded in serving the SpaceTx users.

### Complete 3D support across _starfish_ modules
There are some modules for which 3D support is possible, but not requested by SpaceTx users. We should only add this capability as requested by SpaceTx groups.

### A Quality Control Suite for Spot Finding and Segmentation
Both spot finding and segmentation currently require expert analysis to properly fit parameters. A good QC suite including visualizations of the outcomes would help users with less expertise determine whether they have successfully tuned parameters. This will be useful as _starfish_ begins to capture a wider audience that may include computational users who are more familiar with sequencing experiments. However, our initial users will have adequate expertise and therefore this work can wait.

### Decoupling the back-end (SlicedImage) from ImageStack to enable additional back-end implementations
It would be ideal for us to be able to support different implementations of the SpaceTx Format (e.g. Zarr). A good starting point would be to decouple the ImageStack from the back-end by creating an API. However, the existing implementation is adequate for SpaceTx purposes.

### Tooling for composing codebooks
Users have expressed an interest in programmatically creating codebooks.

### Determine an object model for non-spot features such as cell boundaries and an API for interaction with IntensityTable
We currently use a Label Image to represent which object each pixel of an image corresponds to. This representation does not allow for objects to overlap in space, or for ambiguity in cases where a pixel may correspond to two objects. Also, we currently represent this object as a simple Numpy Array, but it should probably be a class.

### Automation of Parameter Selection
Consortia have stated that they require pipelines whose parameters can be learned directly from data for each assay. It is not clear how to accomplish this, and it is not needed to elicit feedback on _starfish_. However, in the longer term it would be a good idea to pair with the consortia requesting these capabilities to add this functionality to _starfish_.

### Exploration of scaling via GPU computation
Groups (see: CytoKit) are beginning to explore the use of GPU computation to accelerate image-based proteomics analysis. We have been thinking of scale via distributed computation, but it is worth understanding the scale achievable via the GPU as well.


<a name="fn1">1</a>: Missing StarMAP and FISSEQ

<a name="fn2">2</a>: We have implementations for the IntensityTable and ExpressionMatrix but are not satisfied with how we're delivering segmentation as a label image in naked numpy array format (2/3). We haven't written a specification (words, schema that would support a different implementation in, say, R) for any of these (0/3).

<a name="fn3">3</a>: Completed pipelines need a copy number plot with high correlation. It should leverage algorithms that can eventually be implemented in _starfish_ and use numpy/pandas etc.

<a name="fn4">4</a>: At this point in development, we're looking for a solution that the _starfish_ team can run to process a small number of datasets. We are focused on scale-out without an emphasis on cost analysis.

<a name="fn5">5</a>: The converter links currently use version `6.0.0-m3` and will need to be updated when the final version of Bio-Formats 6.0.0 are released.

<a name="fn6">6</a>: The format must support partial IO of image data with at least the granularity of individual channels within fields of view.

<a name="fn7">7</a>: 3d smFISH produces 12 Gb volumes, 3d ISS produces 20 Gb volumes, and 3d MExFISH produces 40+ GB volumes.
