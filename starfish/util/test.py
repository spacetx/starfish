def _data_directory():
    import os
    directories = os.path.dirname(__file__).split()
    return os.path.join(*directories[:-2], 'tests', 'data')


DATA_DIR = _data_directory()
