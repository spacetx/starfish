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
