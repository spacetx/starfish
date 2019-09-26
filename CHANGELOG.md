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
