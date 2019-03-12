## JSON Schemas

Each of the json files that comprise a SpaceTx image fileset can be validated
against one of several jsonschemas ([Draft 4](http://json-schema.org/specification-links.html#draft-4)).

| Schema location                        | Description                                                                            |
|:---------------------------------------|:---------------------------------------------------------------------------------------|
| codebook/codebook.json                 | maps patterns of intensity in the channels and rounds of a field of view to target molecules |
| codebook/codeword.json                 | describes the individiual codes contained in a codebook                                | 
| experiment.json                        | top-level object mapping manifests and codebook together                               |
| extras.json                            | extension point used in multiple schemas for storing key/value pairs                   |
| field_of_view/tiles/indices.json       | describes the categorical indices (channel, round, and z-section) of a tile            |
| field_of_view/tiles/coordinates.json   | physical coordinates of a tile                                                         |
| field_of_view/tiles/tiles.json         | specification of a 2-D image tile                                                      |
| field_of_view/field_of_view.json       | 5-D image consisting of multiple 2-D image tiles                                       |
| fov_manifest.json                      | manifest listing one or more json files that each describe a field of view             |
| version.json                           | general purpose version specification used in multiple schemas                         |

See the API and Usage documentation for how to validate your documents against the jsonschema from your
code or the command-line, respectively.
