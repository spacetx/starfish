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

[0.1.0]: https://github.com/spacetx/starfish/releases/tag/0.1.0
[0.0.36]: https://github.com/spacetx/starfish/releases/tag/0.0.36
[0.0.35]: https://github.com/spacetx/starfish/releases/tag/0.0.35
[0.0.34]: https://github.com/spacetx/starfish/releases/tag/0.0.34
[0.0.33]: https://github.com/spacetx/starfish/releases/tag/0.0.33
