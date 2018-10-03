from starfish import Experiment


def MERFISH(use_test_data: bool=False):
    if use_test_data:
        return Experiment.from_json(
            'https://dmf0bdeheu4zf.cloudfront.net/20180926/MERFISH-TEST/experiment.json')
    return Experiment.from_json(
        'https://dmf0bdeheu4zf.cloudfront.net/20180926/MERFISH/experiment.json')


def allen_smFISH(use_test_data: bool=False):
    return Experiment.from_json(
        'https://dmf0bdeheu4zf.cloudfront.net/20180926/allen_smFISH/experiment.json')


def DARTFISH(use_test_data: bool=False):
    if use_test_data:
        return Experiment.from_json(
            'https://dmf0bdeheu4zf.cloudfront.net/20180926/DARTFISH-TEST/experiment.json')
    return Experiment.from_json(
        'https://dmf0bdeheu4zf.cloudfront.net/20180926/DARTFISH/experiment.json')


def ISS(use_test_data: bool=False):
    if use_test_data:
        return Experiment.from_json(
            'https://dmf0bdeheu4zf.cloudfront.net/20180926/ISS-TEST/experiment.json')
    return Experiment.from_json(
        'https://dmf0bdeheu4zf.cloudfront.net/20180926/ISS/experiment.json')


def osmFISH(use_test_data: bool=False):
    return Experiment.from_json(
        'https://dmf0bdeheu4zf.cloudfront.net/20180926/osmFISH/experiment.json')
