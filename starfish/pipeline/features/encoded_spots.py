import pandas as pd


class EncodedSpots:

    def __init__(self, encoded_spots):
        """

        Parameters
        ----------
        encoded_spots : pd.DataFrame

        """
        # todo typechecking here
        self.data = encoded_spots

    def save(self, output_file_name):
        self.data.to_json(output_file_name, orient='records')

    @classmethod
    def load(cls, json_file):
        cls(pd.read_json(json_file, orient='records'))
