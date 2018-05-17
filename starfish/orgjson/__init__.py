import json
import os

from skimage.external.tifffile import TiffFile
from slicedimage import ImageFormat, ImagePartition, Tile, Writer

from starfish.image import Coordinates, Indices


class OrgJsonCommand:
    @classmethod
    def add_to_parser(cls, subparser_root):
        org_json_command = subparser_root.add_parser(
            "org-json",
            help="Read an org-json file and generate an imagepartition TOC.")
        org_json_command.add_argument("org_json", help="Path to the source org-json file")
        org_json_command.add_argument("--pretty", action="store_true", help="Pretty-print the output file")
        org_json_command.set_defaults(starfish_command=OrgJsonCommand.run_command)

        return org_json_command

    @classmethod
    def run_command(cls, args, print_help=False):
        input_dir = os.path.dirname(args.org_json)
        with open(args.org_json) as fh:
            org_json = json.load(fh)

        dimensions = [Coordinates.X, Coordinates.Y, Indices.HYB, Indices.CH]
        indices = {
            Indices.HYB: org_json['metadata']['num_hybs'],
            Indices.CH: org_json['metadata']['num_chs'],
        }
        all_zs = set()
        for tiledata in org_json['data']:
            if 'z' not in tiledata:
                break
            all_zs.add(tiledata['z'])
        else:
            indices[Indices.Z] = len(all_zs)
            dimensions.append(Indices.Z)

        imagepartition = ImagePartition(
            dimensions,
            indices,
            org_json['metadata']['shape'],
            ImageFormat.find_by_extension(org_json['metadata']['format']),
        )

        for tiledata in org_json['data']:
            indices = {
                Indices.HYB: tiledata['hyb'],
                Indices.CH: tiledata['ch'],
            }
            if 'z' in tiledata:
                indices[Indices.Z] = tiledata['z']

            tile = Tile(
                {
                    Coordinates.X: (0.0, 0.0001),
                    Coordinates.Y: (0.0, 0.0001),
                    Coordinates.Z: (0.0, 0.0001),
                },
                indices,
            )
            srcpath = os.path.join(input_dir, tiledata['file'])
            tile.set_source_fh_contextmanager(
                lambda path=srcpath: open(path, "rb"),
                ImageFormat.TIFF,
            )
            tile._name_or_url = tiledata['file']
            imagepartition.add_tile(tile)

        Writer.write_to_path(
            imagepartition,
            os.path.join(input_dir, "hybridization.json"),
            pretty=args.pretty,
            tile_opener=identity_file_namer,
            tile_writer=null_writer,
        )

        experiment = {
            'version': "0.0.0",
            'hybridization_images': "hybridization.json",
            'auxiliary_images': {},
        }

        for new_key, old_key in (
            ('nuclei', 'dapi'),
            ('dots', 'dots'),
        ):
            org_json_matches = [entry for entry in org_json['aux'] if entry['type'] == old_key]
            if len(org_json_matches) == 0:
                continue
            org_json_entry = org_json_matches[0]
            with TiffFile(os.path.join(input_dir, org_json_entry['file'])) as tiff:
                experiment['auxiliary_images'][new_key] = {
                    'file': org_json_entry['file'],
                    'tile_shape': (tiff.pages[0].image_length, tiff.pages[0].image_width),
                    'tile_format': "TIFF",
                    'coordinates': {
                        'x': (0.0, 0.0001),
                        'y': (0.0, 0.0001),
                    },
                }

        with open(os.path.join(input_dir, "experiment.json"), "w") as fh:
            json.dump(experiment, fh, indent=4 if args.pretty else None)


def identity_file_namer(toc_path, tile, ext):
    class fake_handle(object):
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    return fake_handle(tile._name_or_url)


def null_writer(tile, fh):
    pass
