import pandas as pd


class ValidatedTable:

    required_fields = NotImplemented

    """
    This base class defines common methods for the json outputs of the starfish package, each of which is stored
    internally as a pandas dataframe.

    Methods
    -------
    _validate_table(table: pd.DataFrame, required_fields)
        Each subclass of this base must define required_fields as a class object. When initializing a new object,
        table is checked to verify it contains the required fields
    save(output_file_name: str)
        Save the table to json
    load(json_file: str)
        Load a table from json

    """

    def __init__(self, table: pd.DataFrame, required_fields: set) -> None:
        """

        Parameters
        ----------
        table : pd.DataFrame
            Data to validate
        required_fields : set
            fields table is required to have (class object from implementing subclass)

        """
        self._validate_table(table, required_fields)
        self._data = table

    @property
    def data(self):
        return self._data

    @staticmethod
    def _validate_table(table: pd.DataFrame, required_fields: set):
        missing_fields = required_fields.difference(table.columns)
        if missing_fields:
            raise ValueError(f'input table with columns {table.columns} is missing {missing_fields} required fields')

    def save(self, output_file_name: str) -> None:
        """Save class data to json

        Parameters
        ----------
        output_file_name : str
            Name for output json file

        """
        self.data.to_json(output_file_name, orient='records')

    @classmethod
    def load(cls, json_file: str):
        """Load a ValidatedTable from json

        Parameters
        ----------
        json_file : str
            json file to read

        Returns
        -------
        ValidatedTable :
            Table containing the loaded data

        """

        return cls(pd.read_json(json_file))  # type: ignore # all the subclasses have this signature for constructor.
