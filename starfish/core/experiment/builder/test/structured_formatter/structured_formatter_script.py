# Inplace experiment construction monkeypatches the code destructively.  To isolate these side
# effects, we run the experiment construction in a separate process.

import click
from slicedimage import ImageFormat

from starfish.core.experiment.builder.structured_formatter import format_structured_dataset


@click.command()
@click.argument("image_directory_path", type=click.Path(exists=True, file_okay=False))
@click.argument("coordinates_csv_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_path", type=click.Path(exists=True, file_okay=False))
@click.argument("tile_format", type=click.Choice(
    [imageformat.name for imageformat in list(ImageFormat)]))
@click.argument("in_place", type=bool)
def main(
        image_directory_path: str,
        coordinates_csv_path: str,
        output_path: str,
        tile_format: str,
        in_place: bool,
):
    _tile_format = ImageFormat[tile_format]
    format_structured_dataset(
        image_directory_path,
        coordinates_csv_path,
        output_path,
        _tile_format,
        in_place,
    )


if __name__ == "__main__":
    main()
