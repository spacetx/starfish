from starfish import Experiment


def MERFISH(use_test_data: bool=False):
    if use_test_data:
        return Experiment.from_json(
            'https://d2nhj9g34unfro.cloudfront.net/20181005/MERFISH-TEST/experiment.json')
    return Experiment.from_json(
        'https://d2nhj9g34unfro.cloudfront.net/20181005/MERFISH/experiment.json')


def allen_smFISH(use_test_data: bool=False):
    return Experiment.from_json(
        'https://d2nhj9g34unfro.cloudfront.net/20181005/allen_smFISH/experiment.json')


def DARTFISH(use_test_data: bool=False):
    if use_test_data:
        return Experiment.from_json(
            'https://d2nhj9g34unfro.cloudfront.net/20181005/DARTFISH-TEST/experiment.json')
    return Experiment.from_json(
        'https://d2nhj9g34unfro.cloudfront.net/20181005/DARTFISH/experiment.json')


def ISS(use_test_data: bool=False):
    if use_test_data:
        return Experiment.from_json(
            'https://d2nhj9g34unfro.cloudfront.net/20181005/ISS-TEST/experiment.json')
    return Experiment.from_json(
        'https://d2nhj9g34unfro.cloudfront.net/20181005/ISS/experiment.json')


def osmFISH(use_test_data: bool=False):
    return Experiment.from_json(
        'https://d2nhj9g34unfro.cloudfront.net/20181005/osmFISH/experiment.json')


def BaristaSeq(use_test_data: bool=False) -> Experiment:
    """Loads a BaristaSeq dataset generated from mouse visual cortex. The extracted field of view
    comes from an internal layer of V1 (range: 2-5)

    Parameters
    ----------
    use_test_data : bool
        This parameter is not used for this data type, as there is no data of testing size.

    Returns
    -------
    Experiment
        Experiment containing raw image data
    """
    return Experiment.from_json(
        "https://d2nhj9g34unfro.cloudfront.net/browse/formatted/20181028/"
        "BaristaSeq/cropped_formatted/experiment.json"
    )
