import argparse
from zipfile import ZipFile

from starfish.util.argparse import FsExistsType
import requests
from io import BytesIO


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("zip")
    parser.add_argument("output_dir", type=FsExistsType())
    args = parser.parse_args()

    r = requests.get(args.zip)
    z = ZipFile(BytesIO(r.content))
    z.extractall(args.output_dir)


