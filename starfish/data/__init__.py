from starfish import Experiment


def MERFISH(test_data: bool=False):
    if test_data:
        return Experiment.from_json(
            'https://dmf0bdeheu4zf.cloudfront.net/20180919/MERFISH-TEST/experiment.json')
    return Experiment.from_json(
        'https://dmf0bdeheu4zf.cloudfront.net/20180924/MERFISH/experiment.json')


def allen_smFISH(test_data: bool=False):
    return Experiment.from_json(
        'https://dmf0bdeheu4zf.cloudfront.net/20180919/allen_smFISH/experiment.json')


def DARTFISH(test_data: bool=False):
    if test_data:
        return Experiment.from_json(
            'https://dmf0bdeheu4zf.cloudfront.net/20180919/DARTFISH-TEST/experiment.json')
    return Experiment.from_json(
        'https://dmf0bdeheu4zf.cloudfront.net/20180919/DARTFISH/experiment.json')


def ISS(test_data: bool=False):
    if test_data:
        return Experiment.from_json(
            'https://dmf0bdeheu4zf.cloudfront.net/20180919/ISS-TEST/experiment.json')
    return Experiment.from_json(
        'https://dmf0bdeheu4zf.cloudfront.net/20180919/ISS/experiment.json')


def osmFISH(test_data: bool=False):
    return Experiment.from_json(
        'https://dmf0bdeheu4zf.cloudfront.net/20180924/osmFISH/experiment.json')
