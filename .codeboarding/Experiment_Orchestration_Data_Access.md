```mermaid
graph LR
    Experiment_FOV_Orchestrator["Experiment & FOV Orchestrator"]
    Experiment_Builder_Serializer["Experiment Builder & Serializer"]
    Data_Provider_Tile_Fetching_Layer["Data Provider & Tile Fetching Layer"]
    Metadata_Inference_Formatting_Utility["Metadata Inference & Formatting Utility"]
    Pipeline_Execution_Interface["Pipeline Execution Interface"]
    Metadata_Inference_Formatting_Utility -- "provides metadata to" --> Experiment_Builder_Serializer
    Experiment_Builder_Serializer -- "uses for data retrieval" --> Data_Provider_Tile_Fetching_Layer
    Experiment_FOV_Orchestrator -- "utilizes for lazy loading" --> Data_Provider_Tile_Fetching_Layer
    Experiment_FOV_Orchestrator -- "supplies data to" --> Pipeline_Execution_Interface
    Experiment_Builder_Serializer -- "initializes" --> Experiment_FOV_Orchestrator
    Pipeline_Execution_Interface -- "calls" --> Experiment_FOV_Orchestrator
```

[![CodeBoarding](https://img.shields.io/badge/Generated%20by-CodeBoarding-9cf?style=flat-square)](https://github.com/CodeBoarding/CodeBoarding)[![Demo](https://img.shields.io/badge/Try%20our-Demo-blue?style=flat-square)](https://www.codeboarding.org/diagrams)[![Contact](https://img.shields.io/badge/Contact%20us%20-%20contact@codeboarding.org-lightgrey?style=flat-square)](mailto:contact@codeboarding.org)

## Details

Manages the high-level experiment structure, FOV indexing, and lazy-loading of image data from storage.

### Experiment & FOV Orchestrator
Manages the high-level logical hierarchy (Experiment, Field of View, ImageStack) and provides the primary API for accessing spatially-aware data.


**Related Classes/Methods**:

- `starfish.core.experiment.experiment.Experiment`:213-454
- `starfish.core.experiment.experiment.FieldOfView`:33-194
- `starfish.core.experiment.experiment.AlignedImageStackIterator`:197-210



**Source Files:**

- [`docs/source/conf.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingdocs/source/conf.py)
  - `docs.source.conf.setup` ([L160-L161](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingdocs/source/conf.py#L160-L161)) - Function
- [`notebooks/py/smFISH.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingnotebooks/py/smFISH.py)
  - `notebooks.py.smFISH.processing_pipeline` ([L97-L156](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingnotebooks/py/smFISH.py#L97-L156)) - Function
- [`starfish/core/codebook/_format.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/codebook/_format.py)
  - `starfish.core.codebook._format.DocumentKeys` ([L10-L12](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/codebook/_format.py#L10-L12)) - Class
- [`starfish/core/experiment/experiment.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/experiment.py)
  - `starfish.core.experiment.experiment.FieldOfView` ([L33-L194](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/experiment.py#L33-L194)) - Class
  - `starfish.core.experiment.experiment.FieldOfView.__init__` ([L74-L80](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/experiment.py#L74-L80)) - Method
  - `starfish.core.experiment.experiment.FieldOfView.__repr__` ([L82-L93](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/experiment.py#L82-L93)) - Method
  - `starfish.core.experiment.experiment.FieldOfView.name` ([L96-L97](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/experiment.py#L96-L97)) - Method
  - `starfish.core.experiment.experiment.FieldOfView.image_types` ([L100-L101](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/experiment.py#L100-L101)) - Method
  - `starfish.core.experiment.experiment.FieldOfView.get_image` ([L103-L147](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/experiment.py#L103-L147)) - Method
  - `starfish.core.experiment.experiment.FieldOfView.get_images` ([L149-L194](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/experiment.py#L149-L194)) - Method
  - `starfish.core.experiment.experiment.AlignedImageStackIterator` ([L197-L210](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/experiment.py#L197-L210)) - Class
  - `starfish.core.experiment.experiment.Experiment` ([L213-L454](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/experiment.py#L213-L454)) - Class
  - `starfish.core.experiment.experiment.Experiment.__init__` ([L227-L238](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/experiment.py#L227-L238)) - Method
  - `starfish.core.experiment.experiment.Experiment.__repr__` ([L240-L257](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/experiment.py#L240-L257)) - Method
  - `starfish.core.experiment.experiment.Experiment.from_json` ([L260-L336](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/experiment.py#L260-L336)) - Method
  - `starfish.core.experiment.experiment.Experiment.verify_version` ([L339-L348](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/experiment.py#L339-L348)) - Method
  - `starfish.core.experiment.experiment.Experiment.fov` ([L350-L379](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/experiment.py#L350-L379)) - Method
  - `starfish.core.experiment.experiment.Experiment.fovs` ([L381-L409](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/experiment.py#L381-L409)) - Method
  - `starfish.core.experiment.experiment.Experiment.fovs_by_name` ([L411-L428](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/experiment.py#L411-L428)) - Method
  - `starfish.core.experiment.experiment.Experiment.__getitem__` ([L430-L434](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/experiment.py#L430-L434)) - Method
  - `starfish.core.experiment.experiment.Experiment.keys` ([L436-L438](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/experiment.py#L436-L438)) - Method
  - `starfish.core.experiment.experiment.Experiment.values` ([L440-L442](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/experiment.py#L440-L442)) - Method
  - `starfish.core.experiment.experiment.Experiment.items` ([L444-L446](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/experiment.py#L444-L446)) - Method
  - `starfish.core.experiment.experiment.Experiment.codebook` ([L449-L450](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/experiment.py#L449-L450)) - Method
  - `starfish.core.experiment.experiment.Experiment.extras` ([L453-L454](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/experiment.py#L453-L454)) - Method
- [`starfish/core/image/Filter/_base.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/Filter/_base.py)
  - `starfish.core.image.Filter._base.FilterAlgorithm` ([L8-L13](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/Filter/_base.py#L8-L13)) - Class
  - `starfish.core.image.Filter._base.FilterAlgorithm.run` ([L11-L13](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/Filter/_base.py#L11-L13)) - Method
- [`starfish/core/image/Segment/_base.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/Segment/_base.py)
  - `starfish.core.image.Segment._base.SegmentAlgorithm` ([L8-L18](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/Segment/_base.py#L8-L18)) - Class
  - `starfish.core.image.Segment._base.SegmentAlgorithm.run` ([L11-L18](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/Segment/_base.py#L11-L18)) - Method
- [`starfish/core/image/_registration/ApplyTransform/_base.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/_registration/ApplyTransform/_base.py)
  - `starfish.core.image._registration.ApplyTransform._base.ApplyTransformAlgorithm` ([L8-L13](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/_registration/ApplyTransform/_base.py#L8-L13)) - Class
  - `starfish.core.image._registration.ApplyTransform._base.ApplyTransformAlgorithm.run` ([L11-L13](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/_registration/ApplyTransform/_base.py#L11-L13)) - Method
- [`starfish/core/image/_registration/LearnTransform/_base.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/_registration/LearnTransform/_base.py)
  - `starfish.core.image._registration.LearnTransform._base.LearnTransformAlgorithm` ([L7-L12](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/_registration/LearnTransform/_base.py#L7-L12)) - Class
  - `starfish.core.image._registration.LearnTransform._base.LearnTransformAlgorithm.run` ([L10-L12](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/_registration/LearnTransform/_base.py#L10-L12)) - Method
- [`starfish/core/image/_registration/_format.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/_registration/_format.py)
  - `starfish.core.image._registration._format.DocumentKeys` ([L10-L12](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/_registration/_format.py#L10-L12)) - Class
- [`starfish/core/morphology/Binarize/_base.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/morphology/Binarize/_base.py)
  - `starfish.core.morphology.Binarize._base.BinarizeAlgorithm` ([L8-L13](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/morphology/Binarize/_base.py#L8-L13)) - Class
  - `starfish.core.morphology.Binarize._base.BinarizeAlgorithm.run` ([L11-L13](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/morphology/Binarize/_base.py#L11-L13)) - Method
- [`starfish/core/morphology/Filter/_base.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/morphology/Filter/_base.py)
  - `starfish.core.morphology.Filter._base.FilterAlgorithm` ([L7-L17](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/morphology/Filter/_base.py#L7-L17)) - Class
  - `starfish.core.morphology.Filter._base.FilterAlgorithm.run` ([L10-L17](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/morphology/Filter/_base.py#L10-L17)) - Method
- [`starfish/core/morphology/Merge/_base.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/morphology/Merge/_base.py)
  - `starfish.core.morphology.Merge._base.MergeAlgorithm` ([L8-L18](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/morphology/Merge/_base.py#L8-L18)) - Class
  - `starfish.core.morphology.Merge._base.MergeAlgorithm.run` ([L12-L18](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/morphology/Merge/_base.py#L12-L18)) - Method
- [`starfish/core/morphology/Segment/_base.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/morphology/Segment/_base.py)
  - `starfish.core.morphology.Segment._base.SegmentAlgorithm` ([L7-L12](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/morphology/Segment/_base.py#L7-L12)) - Class
  - `starfish.core.morphology.Segment._base.SegmentAlgorithm.run` ([L10-L12](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/morphology/Segment/_base.py#L10-L12)) - Method
- [`starfish/core/spots/AssignTargets/_base.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/AssignTargets/_base.py)
  - `starfish.core.spots.AssignTargets._base.AssignTargetsAlgorithm` ([L8-L23](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/AssignTargets/_base.py#L8-L23)) - Class
  - `starfish.core.spots.AssignTargets._base.AssignTargetsAlgorithm.run` ([L15-L23](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/AssignTargets/_base.py#L15-L23)) - Method
- [`starfish/core/spots/DecodeSpots/_base.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DecodeSpots/_base.py)
  - `starfish.core.spots.DecodeSpots._base.DecodeSpotsAlgorithm` ([L8-L13](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DecodeSpots/_base.py#L8-L13)) - Class
  - `starfish.core.spots.DecodeSpots._base.DecodeSpotsAlgorithm.run` ([L12-L13](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DecodeSpots/_base.py#L12-L13)) - Method
- [`starfish/core/util/argparse.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/util/argparse.py)
  - `starfish.core.util.argparse.FsExistsType` ([L5-L9](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/util/argparse.py#L5-L9)) - Class
  - `starfish.core.util.argparse.FsExistsType.__call__` ([L6-L9](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/util/argparse.py#L6-L9)) - Method
- [`starfish/data.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/data.py)
  - `starfish.data.MERFISH` ([L4-L27](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/data.py#L4-L27)) - Function
  - `starfish.data.allen_smFISH` ([L30-L53](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/data.py#L30-L53)) - Function
  - `starfish.data.MOUSE_V_HUMAN` ([L56-L69](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/data.py#L56-L69)) - Function
  - `starfish.data.DARTFISH` ([L72-L94](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/data.py#L72-L94)) - Function
  - `starfish.data.ISS` ([L97-L120](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/data.py#L97-L120)) - Function
  - `starfish.data.osmFISH` ([L123-L149](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/data.py#L123-L149)) - Function
  - `starfish.data.BaristaSeq` ([L152-L173](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/data.py#L152-L173)) - Function
  - `starfish.data.ImagingMassCytometry` ([L176-L196](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/data.py#L176-L196)) - Function
  - `starfish.data.SeqFISH` ([L199-L223](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/data.py#L199-L223)) - Function
  - `starfish.data.STARmap` ([L226-L249](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/data.py#L226-L249)) - Function
- [`workflows/wdl/iss_spaceTX/recipe.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingworkflows/wdl/iss_spaceTX/recipe.py)
  - `workflows.wdl.iss_spaceTX.recipe.process_fov` ([L6-L55](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingworkflows/wdl/iss_spaceTX/recipe.py#L6-L55)) - Function


### Experiment Builder & Serializer
Handles the transformation of raw image data into the standardized SpaceTx format, managing JSON manifests and tile layouts.


**Related Classes/Methods**:

- `starfish.core.experiment.builder.builder.write_experiment_json`:303-415
- `starfish.core.experiment.builder.builder.build_image`:149-201
- `starfish.core.experiment.builder.inplace.InplaceWriterContract`:20-31



**Source Files:**

- [`starfish/core/experiment/builder/builder.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/builder.py)
  - `starfish.core.experiment.builder.builder.TileIdentifier` ([L45-L51](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/builder.py#L45-L51)) - Class
  - `starfish.core.experiment.builder.builder.build_irregular_image` ([L54-L146](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/builder.py#L54-L146)) - Function
  - `starfish.core.experiment.builder.builder.build_irregular_image.reducer_to_sets` ([L76-L85](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/builder.py#L76-L85)) - Function
  - `starfish.core.experiment.builder.builder.build_image` ([L149-L201](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/builder.py#L149-L201)) - Function
  - `starfish.core.experiment.builder.builder.write_irregular_experiment_json` ([L204-L300](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/builder.py#L204-L300)) - Function
  - `starfish.core.experiment.builder.builder.write_experiment_json` ([L303-L415](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/builder.py#L303-L415)) - Function
- [`starfish/core/experiment/builder/defaultproviders.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/defaultproviders.py)
  - `starfish.core.experiment.builder.defaultproviders.RandomNoiseTile` ([L16-L39](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/defaultproviders.py#L16-L39)) - Class
  - `starfish.core.experiment.builder.defaultproviders.OnesTile` ([L42-L71](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/defaultproviders.py#L42-L71)) - Class
  - `starfish.core.experiment.builder.defaultproviders.tile_fetcher_factory` ([L74-L97](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/defaultproviders.py#L74-L97)) - Function
  - `starfish.core.experiment.builder.defaultproviders.tile_fetcher_factory.ResultingClass` ([L86-L95](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/defaultproviders.py#L86-L95)) - Class
- [`starfish/core/experiment/builder/inplace.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/inplace.py)
  - `starfish.core.experiment.builder.inplace.InplaceWriterContract` ([L20-L31](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/inplace.py#L20-L31)) - Class
  - `starfish.core.experiment.builder.inplace.InplaceWriterContract.tile_url_generator` ([L21-L22](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/inplace.py#L21-L22)) - Method
  - `starfish.core.experiment.builder.inplace.InplaceWriterContract.write_tile` ([L24-L31](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/inplace.py#L24-L31)) - Method
  - `starfish.core.experiment.builder.inplace.InplaceFetchedTile` ([L34-L49](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/inplace.py#L34-L49)) - Class
- [`starfish/core/experiment/builder/orderediterator.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/orderediterator.py)
  - `starfish.core.experiment.builder.orderediterator.join_axes_labels` ([L7-L30](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/orderediterator.py#L7-L30)) - Function
  - `starfish.core.experiment.builder.orderediterator.ordered_iterator` ([L33-L42](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/orderediterator.py#L33-L42)) - Function
- [`starfish/core/experiment/builder/providers.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/providers.py)
  - `starfish.core.experiment.builder.providers.TileFetcher.get_tile` ([L68-L74](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/providers.py#L68-L74)) - Method
- [`starfish/core/experiment/builder/structured_formatter.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/structured_formatter.py)
  - `starfish.core.experiment.builder.structured_formatter.ExtraPhysicalCoordinatesWarning` ([L33-L37](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/structured_formatter.py#L33-L37)) - Class
  - `starfish.core.experiment.builder.structured_formatter.PhysicalCoordinateNotPresentError` ([L40-L45](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/structured_formatter.py#L40-L45)) - Class
  - `starfish.core.experiment.builder.structured_formatter.InferredTileResult` ([L49-L52](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/structured_formatter.py#L49-L52)) - Class
  - `starfish.core.experiment.builder.structured_formatter.format_structured_dataset` ([L55-L138](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/structured_formatter.py#L55-L138)) - Function
  - `starfish.core.experiment.builder.structured_formatter.infer_stack_structure` ([L141-L164](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/structured_formatter.py#L141-L164)) - Function
  - `starfish.core.experiment.builder.structured_formatter.read_physical_coordinates_from_csv` ([L167-L208](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/structured_formatter.py#L167-L208)) - Function
  - `starfish.core.experiment.builder.structured_formatter.InferredTile` ([L211-L263](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/structured_formatter.py#L211-L263)) - Class
  - `starfish.core.experiment.builder.structured_formatter.InferredTileFetcher` ([L266-L287](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/structured_formatter.py#L266-L287)) - Class
  - `starfish.core.experiment.builder.structured_formatter.InferredTileFetcher.__init__` ([L267-L275](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/structured_formatter.py#L267-L275)) - Method
  - `starfish.core.experiment.builder.structured_formatter.InferredTileFetcher.get_tile` ([L277-L287](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/structured_formatter.py#L277-L287)) - Method
  - `starfish.core.experiment.builder.structured_formatter._convert_str_to_Number` ([L290-L297](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/structured_formatter.py#L290-L297)) - Function
  - `starfish.core.experiment.builder.structured_formatter._parse_coordinates` ([L300-L327](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/structured_formatter.py#L300-L327)) - Function


### Data Provider & Tile Fetching Layer
Provides an abstraction layer for retrieving individual 2D image tiles, decoupling logical structure from physical storage.


**Related Classes/Methods**:

- `starfish.core.experiment.builder.providers.TileFetcher`:63-74
- `starfish.core.experiment.builder.providers.FetchedTile`:12-60
- `starfish.core.imagestack.parser.tileset._parser.SlicedImageTile`:15-76



**Source Files:**

- [`starfish/core/experiment/builder/defaultproviders.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/defaultproviders.py)
  - `starfish.core.experiment.builder.defaultproviders.OnesTile.__init__` ([L47-L49](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/defaultproviders.py#L47-L49)) - Method
  - `starfish.core.experiment.builder.defaultproviders.OnesTile.shape` ([L52-L53](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/defaultproviders.py#L52-L53)) - Method
  - `starfish.core.experiment.builder.defaultproviders.OnesTile.coordinates` ([L56-L61](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/defaultproviders.py#L56-L61)) - Method
  - `starfish.core.experiment.builder.defaultproviders.OnesTile.format` ([L64-L65](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/defaultproviders.py#L64-L65)) - Method
  - `starfish.core.experiment.builder.defaultproviders.OnesTile.tile_data` ([L67-L71](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/defaultproviders.py#L67-L71)) - Method
- [`starfish/core/experiment/builder/inplace.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/inplace.py)
  - `starfish.core.experiment.builder.inplace.InplaceFetchedTile.filepath` ([L38-L40](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/inplace.py#L38-L40)) - Method
  - `starfish.core.experiment.builder.inplace.InplaceFetchedTile.sha256` ([L44-L46](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/inplace.py#L44-L46)) - Method
  - `starfish.core.experiment.builder.inplace.InplaceFetchedTile.tile_data` ([L48-L49](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/inplace.py#L48-L49)) - Method
- [`starfish/core/experiment/builder/providers.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/providers.py)
  - `starfish.core.experiment.builder.providers.FetchedTile` ([L12-L60](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/providers.py#L12-L60)) - Class
  - `starfish.core.experiment.builder.providers.FetchedTile.__init__` ([L16-L17](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/providers.py#L16-L17)) - Method
  - `starfish.core.experiment.builder.providers.FetchedTile.shape` ([L20-L28](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/providers.py#L20-L28)) - Method
  - `starfish.core.experiment.builder.providers.FetchedTile.coordinates` ([L31-L39](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/providers.py#L31-L39)) - Method
  - `starfish.core.experiment.builder.providers.FetchedTile.extras` ([L42-L50](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/providers.py#L42-L50)) - Method
  - `starfish.core.experiment.builder.providers.FetchedTile.tile_data` ([L52-L60](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/providers.py#L52-L60)) - Method
  - `starfish.core.experiment.builder.providers.TileFetcher` ([L63-L74](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/providers.py#L63-L74)) - Class
- [`starfish/core/imagestack/parser/tilefetcher/_parser.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/imagestack/parser/tilefetcher/_parser.py)
  - `starfish.core.imagestack.parser.tilefetcher._parser.TileFetcherImageTile.__init__` ([L18-L28](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/imagestack/parser/tilefetcher/_parser.py#L18-L28)) - Method
  - `starfish.core.imagestack.parser.tilefetcher._parser.TileFetcherImageTile.tile_shape` ([L31-L32](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/imagestack/parser/tilefetcher/_parser.py#L31-L32)) - Method
  - `starfish.core.imagestack.parser.tilefetcher._parser.TileFetcherImageTile.numpy_array` ([L35-L36](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/imagestack/parser/tilefetcher/_parser.py#L35-L36)) - Method
  - `starfish.core.imagestack.parser.tilefetcher._parser.TileFetcherImageTile.coordinates` ([L39-L56](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/imagestack/parser/tilefetcher/_parser.py#L39-L56)) - Method
  - `starfish.core.imagestack.parser.tilefetcher._parser.TileFetcherImageTile.selector` ([L59-L64](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/imagestack/parser/tilefetcher/_parser.py#L59-L64)) - Method
- [`starfish/core/imagestack/parser/tileset/_parser.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/imagestack/parser/tileset/_parser.py)
  - `starfish.core.imagestack.parser.tileset._parser.SlicedImageTile.__init__` ([L22-L33](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/imagestack/parser/tileset/_parser.py#L22-L33)) - Method
  - `starfish.core.imagestack.parser.tileset._parser.SlicedImageTile._load` ([L35-L38](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/imagestack/parser/tileset/_parser.py#L35-L38)) - Method
  - `starfish.core.imagestack.parser.tileset._parser.SlicedImageTile.tile_shape` ([L41-L47](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/imagestack/parser/tileset/_parser.py#L41-L47)) - Method
  - `starfish.core.imagestack.parser.tileset._parser.SlicedImageTile.numpy_array` ([L50-L53](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/imagestack/parser/tileset/_parser.py#L50-L53)) - Method
  - `starfish.core.imagestack.parser.tileset._parser.SlicedImageTile.coordinates` ([L56-L69](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/imagestack/parser/tileset/_parser.py#L56-L69)) - Method
  - `starfish.core.imagestack.parser.tileset._parser.SlicedImageTile.selector` ([L72-L76](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/imagestack/parser/tileset/_parser.py#L72-L76)) - Method
- [`starfish/core/imagestack/physical_coordinates.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/imagestack/physical_coordinates.py)
  - `starfish.core.imagestack.physical_coordinates._get_physical_coordinates_of_z_plane` ([L4-L7](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/imagestack/physical_coordinates.py#L4-L7)) - Function


### Metadata Inference & Formatting Utility
Automates the discovery of dataset structures by parsing file naming conventions or metadata to map to logical axes.


**Related Classes/Methods**:

- `starfish.core.experiment.builder.structured_formatter.format_structured_dataset`:55-138
- `starfish.core.experiment.builder.structured_formatter.InferredTile`:211-263



**Source Files:**

- [`starfish/core/experiment/builder/structured_formatter.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/structured_formatter.py)
  - `starfish.core.experiment.builder.structured_formatter.InferredTile.__init__` ([L214-L224](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/structured_formatter.py#L214-L224)) - Method
  - `starfish.core.experiment.builder.structured_formatter.InferredTile.filepath` ([L227-L228](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/structured_formatter.py#L227-L228)) - Method
  - `starfish.core.experiment.builder.structured_formatter.InferredTile._ensure_tile_loaded` ([L230-L244](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/structured_formatter.py#L230-L244)) - Method
  - `starfish.core.experiment.builder.structured_formatter.InferredTile.sha256` ([L247-L249](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/structured_formatter.py#L247-L249)) - Method
  - `starfish.core.experiment.builder.structured_formatter.InferredTile.shape` ([L252-L254](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/structured_formatter.py#L252-L254)) - Method
  - `starfish.core.experiment.builder.structured_formatter.InferredTile.coordinates` ([L257-L258](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/structured_formatter.py#L257-L258)) - Method
  - `starfish.core.experiment.builder.structured_formatter.InferredTile.tile_data` ([L260-L263](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/structured_formatter.py#L260-L263)) - Method


### Pipeline Execution Interface
The consumption layer where orchestrated experiment data is fed into image processing and spot detection algorithms.


**Related Classes/Methods**:

- `workflows.wdl.iss_published.recipe.process_fov`:9-68
- `starfish.core.spots.DetectPixels.pixel_spot_decoder.PixelSpotDecoder`:14-91



**Source Files:**

- [`starfish/core/image/Filter/gaussian_high_pass.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/Filter/gaussian_high_pass.py)
  - `starfish.core.image.Filter.gaussian_high_pass.GaussianHighPass` ([L17-L127](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/Filter/gaussian_high_pass.py#L17-L127)) - Class
- [`starfish/core/image/Filter/gaussian_low_pass.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/Filter/gaussian_low_pass.py)
  - `starfish.core.image.Filter.gaussian_low_pass.GaussianLowPass` ([L16-L128](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/Filter/gaussian_low_pass.py#L16-L128)) - Class
- [`starfish/core/image/Filter/reduce.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/Filter/reduce.py)
  - `starfish.core.image.Filter.reduce.Reduce` ([L25-L193](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/Filter/reduce.py#L25-L193)) - Class
- [`starfish/core/image/Filter/richardson_lucy_deconvolution.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/Filter/richardson_lucy_deconvolution.py)
  - `starfish.core.image.Filter.richardson_lucy_deconvolution.DeconvolvePSF` ([L17-L212](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/Filter/richardson_lucy_deconvolution.py#L17-L212)) - Class
- [`starfish/core/image/Filter/white_tophat.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/Filter/white_tophat.py)
  - `starfish.core.image.Filter.white_tophat.WhiteTophat` ([L12-L116](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/Filter/white_tophat.py#L12-L116)) - Class
- [`starfish/core/image/_registration/LearnTransform/translation.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/_registration/LearnTransform/translation.py)
  - `starfish.core.image._registration.LearnTransform.translation.Translation` ([L11-L90](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/_registration/LearnTransform/translation.py#L11-L90)) - Class
- [`starfish/core/spots/DecodeSpots/per_round_max_channel_decoder.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DecodeSpots/per_round_max_channel_decoder.py)
  - `starfish.core.spots.DecodeSpots.per_round_max_channel_decoder.PerRoundMaxChannel` ([L12-L66](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DecodeSpots/per_round_max_channel_decoder.py#L12-L66)) - Class
- [`starfish/core/spots/DetectPixels/pixel_spot_decoder.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DetectPixels/pixel_spot_decoder.py)
  - `starfish.core.spots.DetectPixels.pixel_spot_decoder.PixelSpotDecoder` ([L14-L91](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DetectPixels/pixel_spot_decoder.py#L14-L91)) - Class
- [`starfish/core/spots/FindSpots/blob.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/FindSpots/blob.py)
  - `starfish.core.spots.FindSpots.blob.BlobDetector` ([L28-L234](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/FindSpots/blob.py#L28-L234)) - Class
- [`workflows/wdl/iss_published/recipe.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingworkflows/wdl/iss_published/recipe.py)
  - `workflows.wdl.iss_published.recipe.process_fov` ([L9-L68](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingworkflows/wdl/iss_published/recipe.py#L9-L68)) - Function
- [`workflows/wdl/merfish_published/recipe.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingworkflows/wdl/merfish_published/recipe.py)
  - `workflows.wdl.merfish_published.recipe.process_fov` ([L12-L72](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingworkflows/wdl/merfish_published/recipe.py#L12-L72)) - Function




### [FAQ](https://github.com/CodeBoarding/GeneratedOnBoardings/tree/main?tab=readme-ov-file#faq)