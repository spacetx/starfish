## [0.3.3] - 2025-05-09
- Updating CHANGELOG.md for release 0.3.3
- add MacOS to Github actions workflow (#2068)
- Add normalization argument to LearnTransform.Translation with default value None (#2063)
- updating REQUIREMENTS-JUPYTER.txt to resolve h11 dependency vulnerability (#2066)
- fix: blob.py 2d intensities shape and pd.concat instead of append (#2064)
- Clear outputs of SeqFISH notebook and bug fix on SeqFISH.py (#2059)
- removing mistune package from REQUIREMENTS.txt and minor updates to other requirements files (#2054)
- Bump jinja2 from 3.1.5 to 3.1.6 in /requirements (#2053)

## [0.3.2] - 2025-02-16
- Updating CHANGELOG.md for release 0.3.2
- Bump cryptography from 44.0.0 to 44.0.1 in /requirements (#2051)
- rolling back to Sphinx v5 as v6-8 introduce issues on ReadTheDocs with navbar logo and sub-menus (#2050)
- Scikit image update from 0.21 to 0.23+ (#2049)
- Bump jinja2 from 3.1.4 to 3.1.5 in /requirements (#2048)
- Missing pypi description fix by adding pyproject.toml (#2047)
- Bump actions/setup-python from 2 to 5 (#2042)
- Bump actions/cache from 2 to 4 (#2041)
- Bump actions/checkout from 2 to 4 (#2040)
- Create dependabot.yml to auto update Github Action versions (#2039)
- Bump tornado from 6.4.1 to 6.4.2 in /requirements (#2038)
- replace scipy.ndimage.filters with scipy.ndimage for scipy v2 (#2035)
- Bump lxml-html-clean from 0.3.1 to 0.4.0 in /requirements (#2036)

## [0.3.1] - 2024-11-12
- Updating CHANGELOG.md for release 0.3.1
- Adding support to Python 3.12 (#2027)
- Numpy 1.26 update to support Python 3.12 (#2025)
- fixing blob radius calculation in 2d (#2023)
- Release 0.3.0 (#2021)

## [0.3.0] - 2024-10-05
- Dropping support for Python 3.8 and setting Python 3.9 as minimum (#2018)
- Update readthedocs.yml (#2014)
- Scripts and dependencies updates for Python 3.8 - 3.10 (#2009)
- Add min_distance parameter to peak_local_max call (#2008)
- Various Small Fixes/Improvements (#1985)
- Updated seqFISH Decoding Method: CheckAll Decoder (#1978)
- Bump dask from 2021.9.0 to 2021.10.0 in /requirements (#1968)
- Revert "Revert "Fix coords assignment in CombineAdjustFeatures.run (#1965)" (#1966)" (#1967)
- Revert "Fix coords assignment in CombineAdjustFeatures.run (#1965)" (#1966)
- Fix coords assignment in CombineAdjustFeatures.run (#1965)
- Add to_dict and from_dict methods to TransformsList class (#1956)
- fix notebook tests
- Fix check-notebooks makefile command; Add python 3.9 to slow-tests
- Fix Sphinx docs
- Register pytest markers
- Remove is_volume param from testing TrackpyLocalMaxPeakFinder invocation
- Address skimage deprecation of skimage.morphology.watershed
- Address Xarray pending deprecation for GroupBy.apply
- Remove usage of deprecated numpy builtin types like np.bool/np.int
- Add n_processes parameter to FindSpotsAlgorithm.run() abstract method
- Remove deprecated `indices` param for skimage.feature.peak_local_max
- Resolve failure when `peak_local_max()` finds no spots
- Fix failing nearest neighbor trace builder tests
- Fix composite codebook decoder tests
- Change default value of BlobDetector exclude_border parameter to False
- Fix incorrect type annotation and improper `.data()` method call
- Fix failing morphology utility functions
- Fix failing LearnTransform Translation class
- Fix failing Codebook and IntensityTable shape validator
- Fix failing Codebook.decode_per_round_max() test
- Update .gitignore file
- Defer when docker-smoketest github actiosn job is run
- Bump requirements to clear github actions cache
- Fix broken github actions Lint job; Fix broken caching
- Address Mypy type annotation errors
- Update github actions to use caching; Disable Napari tests for now
- Use `python -m pip install ...` to perform pip upgrade
- Replace Travis CI with Github Actions

## [0.2.2] - 2021-04-29
- Updates requirements
- Updates to documentation
- Import match_histograms from skimage.exposure
- Add necessary coords for IntensityTable when using Nearest Neighbors strategy (#1928)
- Fix localmaxpeakfinder spot_props filter (#1839)

## [0.2.1] - 2020-06-08
- Bump napari to 0.3.4 (#1889)
- fix how spot_ids are handled by build_traces_sequential and Label._assign() (#1872)
- reorganized examples gallery and made clarifications to example pipelines and formatting (#1880)
- added image registration tutorial (#1874)
- Add assigning spots to cells docs (#1832)
- Add a Quick Start tutorial (#1869)
- Update starfish installation guide for JOSS submission (#1868)
- Add image segmentation docs (#1821)
- Changing return value of PixelDecoding to DecodedIntensityTable (#1823)
- Ensure that LocalMaxPeakFinder works in 3D (#1822)
- Deprecate is_volume parameter with trackpy (#1820)
- Fix on-demand calculation of BinaryMaskCollection's regionprops (#1819)
- Remove workaround for non-3D images (#1808)
- improve from_code_array validation (#1806)
- Add group_by for tilefetcher-based ImageStack construction (#1796)

## [0.2.0] - 2020-01-31
- Add level_method to the clip filters. (#1758)
- adding method to use installed ilastik instance (#1740)
- Create a TileFetcher-based constructor for ImageStack (#1737)
- adding mouse v human example to starfish.data (#1741)
- adding method to binary mask collection that imports labeled images from external sources like ilastik (#1731)
- Remove starfish.types.Clip (#1729)
- Move watershed segmentation from morphology.Binarize to morphology.Segment (#1720)
- Link to the available datasets in "loading data" section (#1722)
- Document workaround for python3.8 (#1705)
- Wrap skimage's watershed (#1700)
- Add 3D support to target assignment. (#1699)
- Pipeline component and implementation for merging BinaryMaskCollections (#1692)
- Mechanism to reduce multiple masks into one (#1684)

## [0.1.10] - 2019-12-13
- Bump slicedimage to 4.1.1 (#1697)
- Make map/reduce APIs more intuitive (#1686)
- updates roadmap to reflect 2020H1 plans
- adding aws scaling vignette (#1638)
- Use thresholded binarize and mask filtering in existing watershed code. (#1671)
- adding spot ids to pixel results (#1687)
- Implement Labeling algorithms (#1680)
- Thresholded binarize conversion algorithm (#1651)
- Area filter for binary masks (#1673)
- Fix stain generation in watershed (#1670)
- Use the new levels module. (#1669)
- Linear image leveling (#1666)
- add axis labels to display() (#1682)
- Clip method for Richardson Lucy (#1668)
- Filters for mask collections (#1659)
- Provide an apply method to binary mask collections. (#1655)
- adding convience method for slicing codebook data (#1626)
- Fix display tests and code (#1664)
- Additional builders for BinaryMaskCollection (#1637)
- Methods for uncropping binary masks. (#1647)
- Improve coordinate handling code for BinaryMaskCollection and LabelImage (#1632)

## [0.1.9] - 2019-11-18
- Create an ArrayLike type (#1649)
- Verify that binary masks can be generated from empty label images (#1634)
- Add a morphology package to hold BinaryMaskCollection, LabelImage, and their respective operators (#1631)
- fixing travis (#1648)
- Support multiple codewords for the same target (#1646)
- Update data model for BinaryMaskCollection (#1628)
- Test for Codebook.to_json / open_json (#1645)
- Simplify Dockerfile (#1642)
- Switch to version exclusion for scikit-image workaround (#1629)
- Clean up binary mask (#1622)
- adding an extras feild to SpotFindingResults (#1615)
- deleting Decode and Detect modules in lieu of spot finding refactor (#1598)
- Fix install issues (#1641)
- Upgrade to slicedimage 4.1.0 (#1639)
- Update vocabulary for LabelImage I/O operations. (#1630)
- Add a label image data type (#1619)
- Remove deprecated code (#1621)
- fixing bug with codebook.to_json (#1625)
- Don't fill a new ImageStack with NaN (#1609)
- Rename SegmenationMaskCollection to BinaryMaskCollection (#1611)
- Remove hack to force anonymous memory mapping on osx (#1618)

## [0.1.8] - 2019-10-18
- Logging improvements (#1617)
- Make regionprops available per mask (#1610)
- Don't use mypy 0.740 (#1616)
- changing test code to use new spot finding modules (#1597)
- refactoring allen smFish with new spot finding (#1593)
- clean up max projection (#1379)
- Use masked fill to produce labeled images (#1582)
- Replace most instances of starfish.image.Filter.Reduce with imagestack.reduce (#1548)
- implementing starMap spot finding refactor (#1592)
- Add __slots__ to classes that subclass xr.DataArray (#1607)
- Convert SegmentationMaskCollection to a dict-like object (#1579)
- Test case for multiprocessing + imagestack (#1589)
- Masked fill method (#1581)
- Add map/reduce methods to ImageStack (#1539)
- Unify FunctionSource in Map and Reduce (#1540)

## [0.1.7] - 2019-10-09
- ISS refactored with new spot finding path (#1518)
- Fix bugs in per-round-max-decoder (#1602)
- Fix dimension ordering on Codebook and IntensityTable (#1600)
- provanance logging refactor and support for SpotFindingResults (#1517)
- napari 0.2.0 release (#1599)
- starfish.display: unpin napari version, add tests, view masks separately (#1570)
- adding coordinate support to SpotFindingResults (#1516)
- adding new SpotFindingResults data structure and new packages (#1515)

## [0.1.6] - 2019-09-18
- Switch to python multithreading (#1544)
- Don't waste memory/compute in preserve_float_range (#1545)
- Get rid of shared state for LocalMaxPeakFinder (#1541)
- map filter (#1520)
- funcs passed to apply and transform can use positional arguments (#1519)
- import SegmentationMaskCollection in main starfish (#1527)
- Enable Windows builds on master (#1538)
- Throw a warning when the data size is unusual. (#1525)


## [0.1.5] - 2019-08-12
- Update the documentation for data formatters (#1476)
- add ability to convert segmentation masks to a label image
- If in_place=True, we should return None (#1473)
- downgrade pyparsing (#1467)
- fixes unicode in issue template (#1464)
- Adds issue templates (#1460)
- Updating requirements. (#1461)
- Bump to slicedimage 4.0.1 (#1458)
- on-demand loading of data. (#1456)
- Get rid of the check-requirements cron job. (#1448)
- Fixing travis build  (#1457)
- removing duplicate file (#1455)
- Remove Cli (#1444)


## [0.1.4] - 2019-07-16
- Update in-place experiment writing to use the new WriterContract API in slicedimage 4.0.0 (#1447)
- data set formatter with fixed filenames (#1421)

## [0.1.3] - 2019-07-09
- Instantiate the multiprocessing pool using `with` (#1436)
- Slight optimization of pixel decoding  (#1412)
- [easy] point starfish.data.osmFISH() to new dataset (#1425)
- [easy] Warn about the deprecation of the MaxProject filter (#1390)

## [0.1.2] - 2019-06-19
- Refactor reduce to take an optional module and only a function name. (#1386)
- Codify the expectation that in-place experiment construction does not rely on TileFetcher data (#1389)
- Warn and return empty SpotAttributes when PixelDecoding finds 0 spots (#1400)
- updating data.merfish link to full dataset (#1406)
- Rename tile_coordinates to tile_identifier (#1401)
- Support for irregular images in the builder (#1382)
- Fix how we structure the run notebook rules. (#1384)
- updated loading data docs and added image of napari viewer (#1387)
- Format complete ISS experiment and expose in starfish.data (#1316)
- Add concatenate method for ExpressionMatrix (#1381)
- Add TransformsList __repr__ (#1380)
- Fix 3d smFISH notebook as well. (#1385)
- Add custom clip Filter classes (#1376)
- Fix smFISH notebook. (#1383)
- Add Filter.Reduce (general dimension reduction for ImageStack) (#1342)
- Handle denormalized numbers when normalizing intensities/codebooks (#1371)
- TileFetcher formats complete 496 fov MERFISH dataset (#1341)
- Refactor fov.getImage() to fov.getImages() (#1346)
- Add the ability to write labeled experiments (#1374)
- Add inplace TileFetcher module back to public builder API (#1375)
- Always create Z coordinates, even on 4D datasets. (#1358)
- Create an all-purpose ImageStack factory (#1348)
- Remove physical_coordinate_calculator.py (#1352)
- ImageStack parsers should provide coordinates as an array (#1351)
- bump to slicedimage 3.1.1 (#1343)
- Creating a standard starfish.wdl that can be run with any recipe file  (#1364)

## [0.1.1] - 2019-05-16
- [Easy] Fixing spot detection for labeled axes (#1347)
- Schema versioning (#1278)
- Add a missing parameter to wrapper for trackpy local max peak finder (#1300)
- Fix physical coordinate calculator (#1350)
- Fix spot detection for labeled data. (#1349)
- Adding back ability to crop on fov.get_image() (#1329)
- RFC: Base calling filter for in situ sequencing (#1281)
- Preserve subpixel offsets from spot detection (#1330)

## [0.1.0] - 2019-04-23
- public/private separation (#1244)
- Recipe and recipe execution (#1192)
- 3d smFISH notebook (#1238)
- SeqFISH notebook (#1239)
- Adding windows install instructions (#1227)
- vectorize labeling spot lookups (#1215)
- vectorize imagestack -> intensity_table coordinate transfer (#1212)
- Fix the restoration of non-indexed axes. (#1189)
- Allow for intensity tables with labeled axes (#1181)
- ImageStack select on Physical Coordinates (#1147)
- fixing Clip.SCALE_BY_IMAGE (#1193)
- Update BaristaSeq text, fix LinearUnmixing (#1188)
- Update STARmap notebook for SpaceJam (#1199)
- replace label images with segmentation masks (#1135)
- BaristaSeq + Plot tools update (#1171)
- Intensity Table Concat Processing (#1118)

## [0.0.36] - 2019-04-10
- Update strict requirements (#1142)
- High level goal: detect spots should accept imagestacks and not numpy arrays. (#1143)
- Remove cropping from PixelSpotDetector, (#1120)
- Add LocalSearchBlobDetector to support BaristaSeq, SeqFISH, STARmap (#1074)
- Indirect File click types (#1124)
- Move the registration tests next to their sources. (#1134)
- Test to verify that inplace experiment construction works. (#1131)
- Additional support code for building experiments in-place. (#1127)

## [0.0.35] - 2019-04-03
- Transfer physical Coords to Expression Matrix (#965)
- Support for hierarchical directory structures for experiments. (#1126)
- Pipeline Components: LearnTransform and ApplyTransform (#1083)
- Restructure the relationship between PipelineComponent and AlgorithmBase (#1095)


## [0.0.34] - 2019-03-21
- Adding ability to pass aligned group to Imagestack.from_path_or_url (#1069)
- Add Decoded Spot Table (#1087)
- Enable appending to existing napari viewer in display() (#1093)
- Change tile shape to a dict by default (#1072)
- Add ElementWiseMult Filter Pipeline Component (#983)
- Add linear unmixing pipeline component (#1056)
- Spiritual Bag of Images Refactor: Part 1 (#986)
- Add to provenance log   (#968)

## [0.0.33] - 2019.02.14
- Last release without a changelog!

[0.3.3]: https://github.com/spacetx/starfish/releases/tag/0.3.3
[0.3.2]: https://github.com/spacetx/starfish/releases/tag/0.3.2
[0.3.1]: https://github.com/spacetx/starfish/releases/tag/0.3.1
[0.3.0]: https://github.com/spacetx/starfish/releases/tag/0.3.0
[0.2.2]: https://github.com/spacetx/starfish/releases/tag/0.2.2
[0.2.1]: https://github.com/spacetx/starfish/releases/tag/0.2.1
[0.2.0]: https://github.com/spacetx/starfish/releases/tag/0.2.0
[0.1.10]: https://github.com/spacetx/starfish/releases/tag/0.1.10
[0.1.9]: https://github.com/spacetx/starfish/releases/tag/0.1.9
[0.1.8]: https://github.com/spacetx/starfish/releases/tag/0.1.8
[0.1.7]: https://github.com/spacetx/starfish/releases/tag/0.1.7
[0.1.6]: https://github.com/spacetx/starfish/releases/tag/0.1.6
[0.1.5]: https://github.com/spacetx/starfish/releases/tag/0.1.5
[0.1.4]: https://github.com/spacetx/starfish/releases/tag/0.1.4
[0.1.3]: https://github.com/spacetx/starfish/releases/tag/0.1.3
[0.1.2]: https://github.com/spacetx/starfish/releases/tag/0.1.2
[0.1.1]: https://github.com/spacetx/starfish/releases/tag/0.1.1
[0.1.0]: https://github.com/spacetx/starfish/releases/tag/0.1.0
[0.0.36]: https://github.com/spacetx/starfish/releases/tag/0.0.36
[0.0.35]: https://github.com/spacetx/starfish/releases/tag/0.0.35
[0.0.34]: https://github.com/spacetx/starfish/releases/tag/0.0.34
[0.0.33]: https://github.com/spacetx/starfish/releases/tag/0.0.33
