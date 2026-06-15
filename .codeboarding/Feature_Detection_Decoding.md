```mermaid
graph LR
    Spot_Detection_Quantification["Spot Detection & Quantification"]
    Pixel_Level_Decoding_Feature_Integration["Pixel-Level Decoding & Feature Integration"]
    Trace_Assembly_Validation["Trace Assembly & Validation"]
    Spot_Detection_Quantification -- "passes results to" --> Trace_Assembly_Validation
    Trace_Assembly_Validation -- "populates table with data for" --> Pixel_Level_Decoding_Feature_Integration
    Pixel_Level_Decoding_Feature_Integration -- "uses utilities to resolve conflicts" --> Trace_Assembly_Validation
    Pixel_Level_Decoding_Feature_Integration -- "calls" --> Spot_Detection_Quantification
    Trace_Assembly_Validation -- "calls" --> Spot_Detection_Quantification
```

[![CodeBoarding](https://img.shields.io/badge/Generated%20by-CodeBoarding-9cf?style=flat-square)](https://github.com/CodeBoarding/CodeBoarding)[![Demo](https://img.shields.io/badge/Try%20our-Demo-blue?style=flat-square)](https://www.codeboarding.org/diagrams)[![Contact](https://img.shields.io/badge/Contact%20us%20-%20contact@codeboarding.org-lightgrey?style=flat-square)](mailto:contact@codeboarding.org)

## Details

Extracts quantitative signal from processed images, identifying spots or decoding pixel-level signatures.

### Spot Detection & Quantification
Implements computer vision strategies to identify candidate features in image stacks, locating local maxima and extracting physical attributes and multi-channel intensity profiles.


**Related Classes/Methods**:

- `starfish.core.spots.FindSpots.blob.BlobDetector`:28-234
- `starfish.core.spots.FindSpots.local_max_peak_finder.LocalMaxPeakFinder`:29-364
- `starfish.core.spots.FindSpots.spot_finding_utils.measure_intensities_at_spot_locations_across_imagestack`:65-119
- `starfish.core.types._spot_attributes.SpotAttributes`:11-63



**Source Files:**

- [`starfish/core/spots/FindSpots/_base.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/FindSpots/_base.py)
  - `starfish.core.spots.FindSpots._base.FindSpotsAlgorithm` ([L11-L63](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/FindSpots/_base.py#L11-L63)) - Class
  - `starfish.core.spots.FindSpots._base.FindSpotsAlgorithm.run` ([L43-L51](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/FindSpots/_base.py#L43-L51)) - Method
  - `starfish.core.spots.FindSpots._base.FindSpotsAlgorithm._get_measurement_function` ([L54-L63](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/FindSpots/_base.py#L54-L63)) - Method
- [`starfish/core/spots/FindSpots/blob.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/FindSpots/blob.py)
  - `starfish.core.spots.FindSpots.blob.BlobDetector.__init__` ([L68-L92](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/FindSpots/blob.py#L68-L92)) - Method
  - `starfish.core.spots.FindSpots.blob.BlobDetector.image_to_spots` ([L94-L165](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/FindSpots/blob.py#L94-L165)) - Method
  - `starfish.core.spots.FindSpots.blob.BlobDetector.run` ([L167-L234](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/FindSpots/blob.py#L167-L234)) - Method
- [`starfish/core/spots/FindSpots/local_max_peak_finder.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/FindSpots/local_max_peak_finder.py)
  - `starfish.core.spots.FindSpots.local_max_peak_finder.LocalMaxPeakFinder` ([L29-L364](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/FindSpots/local_max_peak_finder.py#L29-L364)) - Class
  - `starfish.core.spots.FindSpots.local_max_peak_finder.LocalMaxPeakFinder.__init__` ([L61-L82](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/FindSpots/local_max_peak_finder.py#L61-L82)) - Method
  - `starfish.core.spots.FindSpots.local_max_peak_finder.LocalMaxPeakFinder._compute_num_spots_per_threshold` ([L84-L143](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/FindSpots/local_max_peak_finder.py#L84-L143)) - Method
  - `starfish.core.spots.FindSpots.local_max_peak_finder.LocalMaxPeakFinder._select_optimal_threshold` ([L145-L188](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/FindSpots/local_max_peak_finder.py#L145-L188)) - Method
  - `starfish.core.spots.FindSpots.local_max_peak_finder.LocalMaxPeakFinder._compute_threshold` ([L190-L210](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/FindSpots/local_max_peak_finder.py#L190-L210)) - Method
  - `starfish.core.spots.FindSpots.local_max_peak_finder.LocalMaxPeakFinder.image_to_spots` ([L212-L310](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/FindSpots/local_max_peak_finder.py#L212-L310)) - Method
  - `starfish.core.spots.FindSpots.local_max_peak_finder.LocalMaxPeakFinder.run` ([L312-L364](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/FindSpots/local_max_peak_finder.py#L312-L364)) - Method
  - `starfish.core.spots.FindSpots.local_max_peak_finder.combine_spot_attributes_by_round_channel` ([L367-L381](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/FindSpots/local_max_peak_finder.py#L367-L381)) - Function
- [`starfish/core/spots/FindSpots/spot_finding_utils.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/FindSpots/spot_finding_utils.py)
  - `starfish.core.spots.FindSpots.spot_finding_utils.measure_intensities_at_spot_locations_in_image` ([L18-L62](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/FindSpots/spot_finding_utils.py#L18-L62)) - Function
  - `starfish.core.spots.FindSpots.spot_finding_utils.measure_intensities_at_spot_locations_in_image.fn` ([L47-L49](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/FindSpots/spot_finding_utils.py#L47-L49)) - Function
  - `starfish.core.spots.FindSpots.spot_finding_utils.measure_intensities_at_spot_locations_across_imagestack` ([L65-L119](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/FindSpots/spot_finding_utils.py#L65-L119)) - Function
- [`starfish/core/spots/FindSpots/trackpy_local_max_peak_finder.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/FindSpots/trackpy_local_max_peak_finder.py)
  - `starfish.core.spots.FindSpots.trackpy_local_max_peak_finder.TrackpyLocalMaxPeakFinder` ([L14-L195](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/FindSpots/trackpy_local_max_peak_finder.py#L14-L195)) - Class
  - `starfish.core.spots.FindSpots.trackpy_local_max_peak_finder.TrackpyLocalMaxPeakFinder.__init__` ([L70-L102](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/FindSpots/trackpy_local_max_peak_finder.py#L70-L102)) - Method
  - `starfish.core.spots.FindSpots.trackpy_local_max_peak_finder.TrackpyLocalMaxPeakFinder.image_to_spots` ([L104-L151](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/FindSpots/trackpy_local_max_peak_finder.py#L104-L151)) - Method
  - `starfish.core.spots.FindSpots.trackpy_local_max_peak_finder.TrackpyLocalMaxPeakFinder.run` ([L153-L195](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/FindSpots/trackpy_local_max_peak_finder.py#L153-L195)) - Method
- [`starfish/core/types/_spot_attributes.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_spot_attributes.py)
  - `starfish.core.types._spot_attributes.SpotAttributes` ([L11-L63](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_spot_attributes.py#L11-L63)) - Class
  - `starfish.core.types._spot_attributes.SpotAttributes.__init__` ([L20-L28](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_spot_attributes.py#L20-L28)) - Method
  - `starfish.core.types._spot_attributes.SpotAttributes.empty` ([L31-L35](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_spot_attributes.py#L31-L35)) - Method
  - `starfish.core.types._spot_attributes.SpotAttributes.combine` ([L38-L42](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_spot_attributes.py#L38-L42)) - Method
  - `starfish.core.types._spot_attributes.SpotAttributes.save_geojson` ([L44-L63](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_spot_attributes.py#L44-L63)) - Method
- [`starfish/core/types/_spot_finding_results.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_spot_finding_results.py)
  - `starfish.core.types._spot_finding_results.PerImageSliceSpotResults` ([L15-L21](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_spot_finding_results.py#L15-L21)) - Class
  - `starfish.core.types._spot_finding_results.SpotFindingResults` ([L24-L240](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_spot_finding_results.py#L24-L240)) - Class


### Pixel-Level Decoding & Feature Integration
Executes decoding logic at the pixel level for multiplexed assays, managing the transition from raw pixel classifications to integrated features and providing Xarray-backed data structures.


**Related Classes/Methods**:

- `starfish.core.spots.DetectPixels.pixel_spot_decoder.PixelSpotDecoder`:14-91
- `starfish.core.intensity_table.intensity_table.IntensityTable`:27-456
- `starfish.core.spots.DetectPixels.combine_adjacent_features.CombineAdjacentFeatures`:81-439
- `starfish.core.intensity_table.decoded_intensity_table.DecodedIntensityTable`:16-191



**Source Files:**

- [`notebooks/py/DARTFISH.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingnotebooks/py/DARTFISH.py)
  - `notebooks.py.DARTFISH.compute_magnitudes` ([L105-L111](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingnotebooks/py/DARTFISH.py#L105-L111)) - Function
- [`starfish/core/intensity_table/decoded_intensity_table.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/intensity_table/decoded_intensity_table.py)
  - `starfish.core.intensity_table.decoded_intensity_table.DecodedIntensityTable` ([L16-L191](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/intensity_table/decoded_intensity_table.py#L16-L191)) - Class
- [`starfish/core/intensity_table/intensity_table.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/intensity_table/intensity_table.py)
  - `starfish.core.intensity_table.intensity_table.IntensityTable._build_xarray_coords` ([L81-L96](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/intensity_table/intensity_table.py#L81-L96)) - Method
  - `starfish.core.intensity_table.intensity_table.IntensityTable.zeros` ([L99-L137](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/intensity_table/intensity_table.py#L99-L137)) - Method
  - `starfish.core.intensity_table.intensity_table.IntensityTable.from_spot_data` ([L140-L198](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/intensity_table/intensity_table.py#L140-L198)) - Method
  - `starfish.core.intensity_table.intensity_table.IntensityTable.get_log` ([L200-L209](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/intensity_table/intensity_table.py#L200-L209)) - Method
  - `starfish.core.intensity_table.intensity_table.IntensityTable.has_physical_coords` ([L212-L214](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/intensity_table/intensity_table.py#L212-L214)) - Method
  - `starfish.core.intensity_table.intensity_table.IntensityTable.to_netcdf` ([L216-L226](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/intensity_table/intensity_table.py#L216-L226)) - Method
  - `starfish.core.intensity_table.intensity_table.IntensityTable.open_netcdf` ([L229-L250](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/intensity_table/intensity_table.py#L229-L250)) - Method
  - `starfish.core.intensity_table.intensity_table.IntensityTable.synthetic_intensities` ([L253-L325](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/intensity_table/intensity_table.py#L253-L325)) - Method
  - `starfish.core.intensity_table.intensity_table.IntensityTable.from_image_stack` ([L328-L399](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/intensity_table/intensity_table.py#L328-L399)) - Method
  - `starfish.core.intensity_table.intensity_table.IntensityTable._process_overlaps` ([L402-L419](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/intensity_table/intensity_table.py#L402-L419)) - Method
  - `starfish.core.intensity_table.intensity_table.IntensityTable.concatenate_intensity_tables` ([L422-L445](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/intensity_table/intensity_table.py#L422-L445)) - Method
  - `starfish.core.intensity_table.intensity_table.IntensityTable.to_features_dataframe` ([L447-L456](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/intensity_table/intensity_table.py#L447-L456)) - Method
- [`starfish/core/spots/DetectPixels/_base.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DetectPixels/_base.py)
  - `starfish.core.spots.DetectPixels._base.DetectPixelsAlgorithm` ([L13-L32](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DetectPixels/_base.py#L13-L32)) - Class
  - `starfish.core.spots.DetectPixels._base.DetectPixelsAlgorithm.run` ([L16-L22](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DetectPixels/_base.py#L16-L22)) - Method
  - `starfish.core.spots.DetectPixels._base.DetectPixelsAlgorithm._get_measurement_function` ([L25-L32](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DetectPixels/_base.py#L25-L32)) - Method
- [`starfish/core/spots/DetectPixels/combine_adjacent_features.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DetectPixels/combine_adjacent_features.py)
  - `starfish.core.spots.DetectPixels.combine_adjacent_features.ConnectedComponentDecodingResult` ([L18-L21](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DetectPixels/combine_adjacent_features.py#L18-L21)) - Class
  - `starfish.core.spots.DetectPixels.combine_adjacent_features.TargetsMap` ([L24-L78](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DetectPixels/combine_adjacent_features.py#L24-L78)) - Class
  - `starfish.core.spots.DetectPixels.combine_adjacent_features.TargetsMap.__init__` ([L26-L43](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DetectPixels/combine_adjacent_features.py#L26-L43)) - Method
  - `starfish.core.spots.DetectPixels.combine_adjacent_features.TargetsMap.targets_as_int` ([L45-L59](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DetectPixels/combine_adjacent_features.py#L45-L59)) - Method
  - `starfish.core.spots.DetectPixels.combine_adjacent_features.TargetsMap.targets_as_str` ([L61-L75](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DetectPixels/combine_adjacent_features.py#L61-L75)) - Method
  - `starfish.core.spots.DetectPixels.combine_adjacent_features.TargetsMap.target_as_str` ([L77-L78](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DetectPixels/combine_adjacent_features.py#L77-L78)) - Method
  - `starfish.core.spots.DetectPixels.combine_adjacent_features.CombineAdjacentFeatures` ([L81-L439](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DetectPixels/combine_adjacent_features.py#L81-L439)) - Class
  - `starfish.core.spots.DetectPixels.combine_adjacent_features.CombineAdjacentFeatures.__init__` ([L83-L110](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DetectPixels/combine_adjacent_features.py#L83-L110)) - Method
  - `starfish.core.spots.DetectPixels.combine_adjacent_features.CombineAdjacentFeatures._intensities_to_decoded_image` ([L113-L150](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DetectPixels/combine_adjacent_features.py#L113-L150)) - Method
  - `starfish.core.spots.DetectPixels.combine_adjacent_features.CombineAdjacentFeatures._calculate_mean_pixel_traces` ([L153-L222](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DetectPixels/combine_adjacent_features.py#L153-L222)) - Method
  - `starfish.core.spots.DetectPixels.combine_adjacent_features.CombineAdjacentFeatures._single_spot_attributes` ([L225-L295](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DetectPixels/combine_adjacent_features.py#L225-L295)) - Method
  - `starfish.core.spots.DetectPixels.combine_adjacent_features.CombineAdjacentFeatures._create_spot_attributes` ([L297-L351](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DetectPixels/combine_adjacent_features.py#L297-L351)) - Method
  - `starfish.core.spots.DetectPixels.combine_adjacent_features.CombineAdjacentFeatures.run` ([L353-L439](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DetectPixels/combine_adjacent_features.py#L353-L439)) - Method
- [`starfish/core/spots/DetectPixels/pixel_spot_decoder.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DetectPixels/pixel_spot_decoder.py)
  - `starfish.core.spots.DetectPixels.pixel_spot_decoder.PixelSpotDecoder.__init__` ([L36-L47](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DetectPixels/pixel_spot_decoder.py#L36-L47)) - Method
  - `starfish.core.spots.DetectPixels.pixel_spot_decoder.PixelSpotDecoder.run` ([L49-L91](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DetectPixels/pixel_spot_decoder.py#L49-L91)) - Method


### Trace Assembly & Validation
Orchestrates the assembly of traces from detected spots, ensures schema conformance, and resolves spatial overlaps between adjacent image tiles.


**Related Classes/Methods**:

- `starfish.core.spots.DecodeSpots.trace_builders.build_traces_sequential`:41-75
- `starfish.core.types._spot_finding_results.SpotFindingResults`:24-240
- `starfish.core.types._validated_table.ValidatedTable`:6-80
- `starfish.core.intensity_table.overlap.find_overlaps_of_xarrays`:50-81



**Source Files:**

- [`starfish/core/intensity_table/overlap.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/intensity_table/overlap.py)
  - `starfish.core.intensity_table.overlap.Area` ([L10-L47](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/intensity_table/overlap.py#L10-L47)) - Class
  - `starfish.core.intensity_table.overlap.Area.__init__` ([L15-L19](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/intensity_table/overlap.py#L15-L19)) - Method
  - `starfish.core.intensity_table.overlap.Area.__eq__` ([L21-L25](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/intensity_table/overlap.py#L21-L25)) - Method
  - `starfish.core.intensity_table.overlap.Area._overlap` ([L28-L34](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/intensity_table/overlap.py#L28-L34)) - Method
  - `starfish.core.intensity_table.overlap.Area.find_intersection` ([L37-L47](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/intensity_table/overlap.py#L37-L47)) - Method
  - `starfish.core.intensity_table.overlap.find_overlaps_of_xarrays` ([L50-L81](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/intensity_table/overlap.py#L50-L81)) - Function
  - `starfish.core.intensity_table.overlap.remove_area_of_xarray` ([L84-L104](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/intensity_table/overlap.py#L84-L104)) - Function
  - `starfish.core.intensity_table.overlap.sel_area_of_xarray` ([L107-L126](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/intensity_table/overlap.py#L107-L126)) - Function
  - `starfish.core.intensity_table.overlap.take_max` ([L129-L161](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/intensity_table/overlap.py#L129-L161)) - Function
- [`starfish/core/spots/DecodeSpots/simple_lookup_decoder.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DecodeSpots/simple_lookup_decoder.py)
  - `starfish.core.spots.DecodeSpots.simple_lookup_decoder.SimpleLookupDecoder` ([L10-L53](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DecodeSpots/simple_lookup_decoder.py#L10-L53)) - Class
  - `starfish.core.spots.DecodeSpots.simple_lookup_decoder.SimpleLookupDecoder.__init__` ([L23-L24](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DecodeSpots/simple_lookup_decoder.py#L23-L24)) - Method
  - `starfish.core.spots.DecodeSpots.simple_lookup_decoder.SimpleLookupDecoder.run` ([L26-L53](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DecodeSpots/simple_lookup_decoder.py#L26-L53)) - Method
- [`starfish/core/spots/DecodeSpots/trace_builders.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DecodeSpots/trace_builders.py)
  - `starfish.core.spots.DecodeSpots.trace_builders.build_spot_traces_exact_match` ([L16-L38](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DecodeSpots/trace_builders.py#L16-L38)) - Function
  - `starfish.core.spots.DecodeSpots.trace_builders.build_traces_sequential` ([L41-L75](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DecodeSpots/trace_builders.py#L41-L75)) - Function
  - `starfish.core.spots.DecodeSpots.trace_builders.build_traces_nearest_neighbors` ([L78-L108](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DecodeSpots/trace_builders.py#L78-L108)) - Function
- [`starfish/core/spots/DecodeSpots/util.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DecodeSpots/util.py)
  - `starfish.core.spots.DecodeSpots.util._match_spots` ([L12-L56](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DecodeSpots/util.py#L12-L56)) - Function
  - `starfish.core.spots.DecodeSpots.util._build_intensity_table` ([L59-L124](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DecodeSpots/util.py#L59-L124)) - Function
  - `starfish.core.spots.DecodeSpots.util._merge_spots_by_round` ([L127-L160](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spots/DecodeSpots/util.py#L127-L160)) - Function
- [`starfish/core/types/_decoded_spots.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_decoded_spots.py)
  - `starfish.core.types._decoded_spots.DecodedSpots.__init__` ([L15-L23](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_decoded_spots.py#L15-L23)) - Method
  - `starfish.core.types._decoded_spots.DecodedSpots.save_csv` ([L25-L26](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_decoded_spots.py#L25-L26)) - Method
  - `starfish.core.types._decoded_spots.DecodedSpots.load_csv` ([L29-L30](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_decoded_spots.py#L29-L30)) - Method
- [`starfish/core/types/_spot_finding_results.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_spot_finding_results.py)
  - `starfish.core.types._spot_finding_results.SpotFindingResults.__init__` ([L31-L62](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_spot_finding_results.py#L31-L62)) - Method
  - `starfish.core.types._spot_finding_results.SpotFindingResults.__setitem__` ([L64-L75](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_spot_finding_results.py#L64-L75)) - Method
  - `starfish.core.types._spot_finding_results.SpotFindingResults.__getitem__` ([L77-L89](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_spot_finding_results.py#L77-L89)) - Method
  - `starfish.core.types._spot_finding_results.SpotFindingResults.items` ([L91-L95](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_spot_finding_results.py#L91-L95)) - Method
  - `starfish.core.types._spot_finding_results.SpotFindingResults.keys` ([L97-L101](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_spot_finding_results.py#L97-L101)) - Method
  - `starfish.core.types._spot_finding_results.SpotFindingResults.values` ([L103-L107](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_spot_finding_results.py#L103-L107)) - Method
  - `starfish.core.types._spot_finding_results.SpotFindingResults.save` ([L109-L146](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_spot_finding_results.py#L109-L146)) - Method
  - `starfish.core.types._spot_finding_results.SpotFindingResults.load` ([L149-L194](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_spot_finding_results.py#L149-L194)) - Method
  - `starfish.core.types._spot_finding_results.SpotFindingResults.round_labels` ([L197-L201](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_spot_finding_results.py#L197-L201)) - Method
  - `starfish.core.types._spot_finding_results.SpotFindingResults.ch_labels` ([L204-L209](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_spot_finding_results.py#L204-L209)) - Method
  - `starfish.core.types._spot_finding_results.SpotFindingResults.count_total_spots` ([L211-L218](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_spot_finding_results.py#L211-L218)) - Method
  - `starfish.core.types._spot_finding_results.SpotFindingResults.get_physical_coord_ranges` ([L221-L226](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_spot_finding_results.py#L221-L226)) - Method
  - `starfish.core.types._spot_finding_results.SpotFindingResults.log` ([L229-L240](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_spot_finding_results.py#L229-L240)) - Method
- [`starfish/core/types/_validated_table.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_validated_table.py)
  - `starfish.core.types._validated_table.ValidatedTable` ([L6-L80](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_validated_table.py#L6-L80)) - Class
  - `starfish.core.types._validated_table.ValidatedTable.__init__` ([L26-L38](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_validated_table.py#L26-L38)) - Method
  - `starfish.core.types._validated_table.ValidatedTable.data` ([L41-L42](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_validated_table.py#L41-L42)) - Method
  - `starfish.core.types._validated_table.ValidatedTable._validate_table` ([L45-L51](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_validated_table.py#L45-L51)) - Method
  - `starfish.core.types._validated_table.ValidatedTable.save` ([L53-L62](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_validated_table.py#L53-L62)) - Method
  - `starfish.core.types._validated_table.ValidatedTable.load` ([L65-L80](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_validated_table.py#L65-L80)) - Method




### [FAQ](https://github.com/CodeBoarding/GeneratedOnBoardings/tree/main?tab=readme-ov-file#faq)