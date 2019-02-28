# _starfish_ Roadmap
This document describes the features and timeline for current work on the _starfish_ package.

The purpose of this document is to describe the deliverables that we intend to deliver by June 2019. To accomplish this, we first outline our primary [use cases](#use-cases). With these to guide us, we then define milestones for releases that support these needs [**what we need to build**](#what-we-need-to-build:-starfish-0.1.0), and candidate features which we may consider building in the future [**what we can build later**](candidates-features-for-0.2.0+).

## Use cases
_starfish_ is being developed to support the data processing and benchmarking needs of the SpaceTx consortium and the Chan Zuckerberg Biohub. We have identified a set of use cases from these groups that _starfish_ will support. Feedback from these users will set the course for future work on _starfish_, beyond the milestones identified here.

### Local processing of data
Some image-based transcriptomics asays have moderate processing requirements. Users of these assays would like to be able to use _starfish_ to process multiple fields of view of data on their local machines to enable further parameter tuning and spot-checking of results. To support this, we will define a Python API for running _starfish_ on local machines that processes experiments of arbitrary numbers of fields of view, logs, stores data provenance, is multi-processing enabled. Users will also be able to run _starfish_ pipelines through a command-line interface (CLI).

### Parameter selection on single fields of view
Image-based transcriptomics workflows are highly dependent upon the tissue, organism, and probes being assayed. Researchers typically select image processing parameters by analyzing and optimizing parameter selection in a single field of view. In support of this use case, _starfish_ must enable users to process data for any SpaceTx assay on the local computer of their choosing and interact with visualization tools necessary to evaluate parameter decisions. It must support this functionality for users that are not proficient in Python.

### Scalable processing of SpaceTx assays
It may not always be possible or time efficient to process an entire experiment on a local machine, therefore _starfish_ must be able to process a complete experiment consisting of hundreds of fields of view that are comprised of up to 10s of terabytes of 2-d TIFF images. For the SpaceTx consortium project, the _starfish_ team agreed to process the data for each of the data contributors, so the _starfish_ team will define a "Scalable Pipeline Runner" that will run _starfish_ pipelines on a parallel computing infrastructure of our choice. We will expose tooling to explore the output formats on a personal computer of our contributor's choosing.

### Scientific assessment of analyzed SpaceTx data: cell type mapping
The SpaceTx consortium Working Group 5 is devoted to mapping the cells in Cell x Gene tables into the transcriptomic types/clusters delivered by Working Group 2.  _starfish_ will define standard output formats for external (non-_starfish_) calculation of these cell types and subsequent incorporation of cell type information as metadata in the Cell x Gene tables.

### Scientific assessment of analyzed SpaceTx data: methods comparison and meta analysis
The SpaceTx consortium Working Group 6 is devoted to comparing the results of each method and assessing their relative strengths and weaknesses. These are largely scientific analyses enabled by the standardization of each assay's outputs by _starfish_.  Because each assay outputs an IntensityTable, Segmentation Mask, and Expression Matrix, it will be simple to label cells or spots with anatomical information represented as named 2D polygons.

By enabling the creation of standardized, highly compressed outputs that are compatible with both R and Python, _starfish_ will enable working Group 6 to ask questions across all methods like "how many cells/mm2 belonging to the Sst-Chodl cluster were found in layer 4 of mouse primary visual cortex?"  or "Within the inhibitory cell class, what is the abundance and distribution of Pvalb+ and VIP+ cells in human cortex?"

Finally, with a standard format to save and load Cell x Gene tables, visualization of this data will be straightforward for analysts in contributor labs and Working Group 6 members.

## What is _starfish_?
We aim for _starfish_ to be a comprehensive platform for the processing of image-based transcriptomics experiments. It should be intuitive to use, fast, and should not require extensive experience with python programming. However, at least during the initial development period, _starfish_ will require domain experience in image processing, as our target users are experts in image generation and processing.

### A General Format and Object Model for Image-based Transcriptomics
_starfish_ defines a general set of data formats (and paired python object implementations) that (1) enable efficient storage and access for a related set of images broken up into single fields of view<sup>[6](#fn6)</sup>, (2) enable storage and utilization of a codebook, which defines how sets of images are combined to decode spot patterns into biological targets, and (3) defines a pipeline recipe which can be interpreted by _starfish_'s pipeline runner to process an experiment.

Currently, starfish needs images in SpaceTx Format must either be pre-aligned, such that the x-y location of each tile in a Field of View is the same. Image data that does not adhere to this requirement will not be processable with _starfish_. See [Candidate features for future releases](#candidate-features-for-future-releases) for more discussion of future directions here.

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

## Outstanding work

### A General Format and Object Model for Image-based Transcriptomics

#### Format specification for _starfish_ outputs, IntensityTable and ExpressionMatrix:
These two objects currently lack format specifications, which has confused some users. A specification for these formats will help computational users to understand how to consume our data formats, and will help facilitate their conversion for use in other languages. For users who are comfortable with python but not necessarily with xarray, we will build in a series of `to_array` and `to_pandas` options.

#### _starfish_ must enable processing of fields of view by round
Sequential smFISH assays that capture volumetric images produce fields of view that are too large for personal computers<sup>[7](#fn7)</sup>. However, if we are able to break up the processing of this data by round, we could hit our scale targets. Second, multiplexed assays (e.g. MERFISH, SeqFISH, ISS) require all rounds and channels to be loaded into memory to decode. In both cases, we should derisk decoding of these assays by understanding the maximum ImageStack size we can support before we need to implement solutions to manage memory consumption.

### A Modular Library of Pipeline Components

#### Documentation
We need to add basic documentation that describes how to use each component, and how to fit parameters. This should be adequate for a skilled computational user to pick up _starfish_ and use it to make a pipeline. _starfish_ should also clearly document how a developer can contribute code they need for their analyses to the project. We will need to source feedback from users on what parts of the documentation provoke confusion.

#### Image re-scaling & normalization
Several approaches normalize images to overcome different channel intensities or round biases. We should implement a pipeline component to support this that samples from images within or across fields of view. Determine the minimum amount of data to sample from images across the experiment to equalize their intensities.

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

## Deliverables, milestones, and timeline

The features that meet these needs can be broken down into 9 categories, each of which centers on a user need

1. Library is usable by SpaceTx for generating pipeline recipes and processing their data on individual machines
2. Documentation supports new users to learn to use _starfish_ without our involvement
3. _starfish_ facilitates interaction with external visualization software (Napari, FIJI) to enable users to tune image processing parameters
4. Library can be used to reproduce collaborator's results for single fields of view
5. The _starfish_ team can run _starfish_ at scale to process the initial set of SpaceTx datasets
6. Experiment-scale output formats can be loaded on local computers and are adequately expressive to enable QC workflows and specified by our users
7. _starfish_ output formats enable external annotation to place the data in anatomical and cell type context and allows comparison across methods
8. Data is easily converted into SpaceTx format from common imaging formats
9. _starfish_ references example data for each supported assay that can be used by researchers to learn starfish and for scientific comparison of supported assays

To build these feature, we have two major phases of development in the first half of 2019, each culminating in minor releases of _starfish_. The first release (0.1.0) will support the needs of SpaceTx members to be able to create _starfish_ pipelines, while the second (0.2.0) will permit running these pipelines at scale.

- 0.1.0 (April 2019): SpaceTx users are able to define _starfish_ pipelines and run them locally
- 0.2.0 (June 2019): Insights can be gained from SpaceTx pipelines run at scale

A more detailed description of the features of each of these releases follows.

### 0.1.0  (April 2019): SpaceTx users are able to define _starfish_ pipelines and run them locally

#### Library is usable by SpaceTx for generating pipeline recipes and processing their data on individual machines
1. Library is easily installed on operating systems used by SpaceTx users
2. Basic for-loop implementation of a pipeline runner to de-risk scale problems that result from processing multiple fields of view, then integrating results for biological analysis
3. Explicit definition/specification of a pipeline recipe
4. API for local processing multiple FOVs for each SpaceTx group given data in SpaceTx format and a pipeline recipe
5. Translate each example pipeline into an example pipeline recipe
6. Visualization tooling to enable parameter selection on small datasets to tweak pipeline recipes
7. API leverages local parallelism

#### Documentation such that new users can learn to use it without our involvement
1. Public and Private API is fully typed and documented
2. Purpose of each pipeline component is described, and guidance on parameter tuning, if necessary, is provided
3. There are examples of formatting data in SpaceTx format
4. There are examples of fully-worked pipelines
5. There are examples of the workflow for creating new pipelines
7. Public API is stable

#### _starfish_ facilitates easy interaction with external visualization software (Napari, FIJI) to enable users to tune image processing parameters
1. _starfish_ has a method to visualize an ImageStack in Napari
2. _starfish_ has a method to plot spots from an IntensityTable onto an ImageStack in Napari
3. _starfish_ can dump ImageStacks for viewing 2D TIFFs in FIJI

#### Library that can be used to reproduce collaborator's results for single fields of view
2. Create a data conversion tool to wrangle contributed data into SpaceTx format (0/10)
3. Specify an input file format and corresponding object model that supports each assay type (8/10)

### 0.2.0 (June 2019): Insights can be gained from SpaceTx pipelines run at scale

#### Library that can be used to reproduce collaborator's results for single fields of view
Where parenthetical fractions are listed, they represent progress towards our goals at the time this roadmap was written.

1. Obtain example data, pipelines, and results from each SpaceTx group (9/10)<sup>[1](#fn1)</sup>
4. Specify output file format specifications for detected spots, gene expression matrices, and detected objects (e.g. cells) (0/3), and corresponding object models  (2/3)<sup>[2](#fn2)</sup>
5. Implementation, in python, of a single-fov proof of concept pipeline that closely matches the processing results of data generators for 9 SpaceTx groups. This is computational biology work to understand the characteristics of the data & assay, and can be done outside _starfish_ as necessary to identify its requirements. (3/10)<sup>[3](#fn3)</sup>
6. Implementation, in _starfish_, of the same single-FOV pipelines above (3/10)

#### A way to run _starfish_ at scale to process SpaceTx datasets
1. _starfish_-based solution to process multiple FOVs at scale tied to a specific infrastructure (Not designed to be run by SpaceTx users on their hardware) (0/9)<sup>[4](#fn4)</sup>

#### Experiment-scale output formats can be loaded on local computers and are adequately expressive to enable QC workflows specified by our users. These output formats enable scientific investigation of the data.
1. _starfish_ is used to process the three example datasets that we have access to:
    1. ISS Breast cancer. 16 FOVs with shape (4, 4, 1, 1024, 1024),
    2. MERFISH U2-OS cells. 400 FOVs with shape (8, 2, 1, 2048, 2048),
    3. osmFISH Visual Cortex. ~100s FOVs with shape (13, 3, 45, 2048, 2048)
2. Outputs from Fields of view can be combined into a single output format
3. Output formats can be loaded and are performant
4. Processed data can be used to recapitulate results

#### _starfish_ output formats enable external annotation to place the data in anatomical and cell type context and allow comparison of across methods
1. Processed Experiments (all FOVs for a given dataset) can produce Cell x Gene tables with each cell in the physical coordinates of the sample
2. 2D polygon annotations of anatomical structures in physical coordinates can be saved in a standardized format and membership of each cell in annotation structures can be added as metadata to the Cell x Gene table
3. Cell x Gene table can be easily imported by computational biologists (outside of _starfish_) to determine cell type or cluster membership for each cell.  This cluster membership can include probabilistic membership to multiple clusters and is added to the Cell x Gene table as metadata

#### A tool to convert image data into SpaceTx Format
1. CLI and API (Java) can parse [supported file formats](https://docs.openmicroscopy.org/bio-formats/6.0.0-m3/supported-formats.html) into SpaceTx-formatted datasets<sup>[5](#fn5)</sup>
2. Installation of the tool can be accomplished by downloading an archive with a launcher script or by using pre-built docker images
3. Users and/or data wranglers can point the tool at one of the [stated files](https://docs.openmicroscopy.org/bio-formats/6.0.0-m3/formats/dataset-table.html) for each FOV to generate 2D TIFFs as well as the necessary SpaceTX JSON files
4. In the case that multiple independent images (a “series”) are present in a single dataset, the user must specify which offset to use
5. Multiple datasets can be passed in which case each in order is considered a separate field of view
5. The starting index of the field of view may be changed from the default of zero
6. If a codebook exists at the time of creation, it can be passed to the tool for inclusion in the JSON. Otherwise, a dummy codebook will be created


## Candidate features for future releases

There are many other features that are good candidates for future releases. We discuss these features, and why we believe they can be built at a later point.

### Hardware- and Acquisition-related corrections

We lack general purpose solutions for registration that fulfill requirements for ISS (the Harris version), smFISH, & MERFISH. Additionally, expansion microscopy requires non-affine thin-plate spline registration. Registration is out of scope for the timeline expected by SpaceTx to have all the data processed.

In order to benchmark datasets, all contributors will need to provide pre-processed images before uploading. This will help us focus on _starfish_'s key value proposition and meet our goal of comparing the results from all the methods using standard file formats, and scalable, reproducible image processing workflows.

A key challenge in developing generalized solutions to registration and other image processing corrections is that the necessary corrections are highly specific to the signal acquisition challenges each data generator faces. Furthermore, proper image pre-processing is critical to the success of the downstream pipeline components that we're building in the roadmap defined above. Because data generators understand their hardware, chemistry, and samples much better than we do, for now, they are best able to apply their solutions before the data goes into _starfish_ pipelines.

In the future, however, we may be able to facilitate users incorporating their solutions or general purpose solutions into _starfish_ pipelines. There are three categories of pre-processing that are worth considering.

#### Affine Registration
For the SpaceTx pilot project, it is adequate to apply pre-computed transformations or leverage already-registered data. Later, when we want to support groups to use _starfish_ for their own research, it will be important to implement affine registration, as otherwise _starfish_ serves only part of their use cases, and they are incentivized to continue running their existing pipelines.

#### Non-Rigid Registration
Non-Rigid registration approaches are very specific to the types of deformations that the tissue suffers, and have variable forms. While we don't believe that these transformations are sufficiently broad in use to implement them in _starfish_, we will investigate defining an API for arbitrary transformations that could allow users to incorporate these approaches into _starfish_ pipelines.

#### Fixing Chromatic Aberration
Chromatic aberrations, including non-uniform differences in illumination, cross-talk between channels, pincushioning, radial transformations, and channel drift tend to be specific to the assay and microscope, making creation of a general solution complicated. Additionally, these problems are often fixed early in pipelines so it is relatively easy to extricate this step from the pipeline. For now, _starfish_ will expect data that has already been corrected, but helping users apply these corrections as part of a _starfish_ pipeline is a good candidate feature for the future. Note that Starfish does enable some very basic tools to learn and apply similarity transformations and can apply uniform intensity normalizations, so some simpler chromatic aberrations may be correctible with _starfish_. If uncertain if your use case is currently supported, please reach out to package maintainers.

### Updating proof-of-concept pipelines with new modules
As work is completed on channel scaling and registration, we will have an opportunity to revisit example pipelines, such as the MERFISH pipeline.

### A Complete Testing Framework
After succeeding in meeting the needs of SpaceTx users, we will have an opportunity to solicit a larger base of users and build a community of developers. We should continue to design tests as needed, and later we should devote some effort to ensuring that our testing suite is robust enough to support contributions from non-core developers without risking the introduction of serious errors.

### Complete 3D support across _starfish_ modules
There are some modules for which 3D support is possible, but not requested by SpaceTx users. If such capability is identified as a need, this will be a good feature to explore.

### A Quality Control Suite for Spot Finding and Segmentation
Both spot finding and segmentation currently require expert analysis to properly fit parameters. A good QC suite including visualizations of the outcomes would help users with less expertise determine whether they have successfully tuned parameters. This will be useful as _starfish_ begins to capture a wider audience that may include computational users who are more familiar with sequencing experiments. However, our initial users have adequate expertise in this domain, making this low priority in the near term.

### Decoupling the back-end (SlicedImage) from ImageStack to enable additional back-end implementations
It would be ideal for us to be able to support different implementations of the SpaceTx Format (e.g. Zarr). A good starting point would be to decouple the ImageStack from the back-end by creating an API. However, the existing implementation is adequate for SpaceTx purposes.

### Tooling for composing codebooks
Users have expressed an interest in programmatically creating codebooks.

### Determine an object model for non-spot features such as cell boundaries and an API for interaction with IntensityTable
We currently use a Label Image to represent which object each pixel of an image corresponds to. This representation does not allow for objects to overlap in space, or for ambiguity in cases where a pixel may correspond to two objects. Also, we currently represent this object as a simple Numpy Array, but it should probably be a class.

### Automation of Parameter Selection
Some users have stated that they require pipelines whose parameters can be learned directly from data for each assay. It is not clear how to accomplish this, and it is not needed to elicit feedback on _starfish_. However, in the longer term it would be a good idea to work with those users requesting these capabilities to add this functionality to _starfish_.

### Exploration of scaling via GPU computation
Groups (see: CytoKit) are beginning to explore the use of GPU computation to accelerate image-based proteomics analysis. We have been thinking of scale via distributed computation, but it is worth understanding the scale achievable via the GPU as well.


<a name="fn1">1</a>: Missing StarMAP and FISSEQ

<a name="fn2">2</a>: We have implementations for the IntensityTable and ExpressionMatrix but are not satisfied with how we're delivering segmentation as a label image in naked numpy array format (2/3). We haven't written a specification (words, schema that would support a different implementation in, say, R) for any of these (0/3).

<a name="fn3">3</a>: Completed pipelines need a copy number plot with high correlation. It should leverage algorithms that can eventually be implemented in _starfish_ and use numpy/pandas etc.

<a name="fn4">4</a>: At this point in development, we're looking for a solution that the _starfish_ team can run to process a small number of datasets. We are focused on scale-out without an emphasis on cost analysis.

<a name="fn5">5</a>: The converter links currently use version `6.0.0-m3` and will need to be updated when the final version of Bio-Formats 6.0.0 are released.

<a name="fn6">6</a>: The format must support partial IO of image data with at least the granularity of individual channels within fields of view.

<a name="fn7">7</a>: 3d smFISH produces 12 Gb volumes, 3d ISS produces 20 Gb volumes, and 3d MExFISH produces 40+ GB volumes.
