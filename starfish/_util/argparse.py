import argparse
import os


class FsExistsType:
    def __call__(self, prospective_dir):
        if not os.path.exists(prospective_dir):
            raise argparse.ArgumentTypeError("{0} does not exist".format(prospective_dir))
        return prospective_dir
