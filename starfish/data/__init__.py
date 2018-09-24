from starfish import Experiment


def MERFISH():
    return Experiment.from_json('https://dmf0bdeheu4zf.cloudfront.net/20180919/MERFISH/experiment.json')

def allen_smFISH():
    return Experiment.from_json('https://dmf0bdeheu4zf.cloudfront.net/20180919/allen_smFISH/experiment.json')

def DARTFISH():
    return Experiment.from_json('https://dmf0bdeheu4zf.cloudfront.net/20180919/DARTFISH/experiment.json')

def ISS():
    return Experiment.from_json('https://dmf0bdeheu4zf.cloudfront.net/20180919/ISS/experiment.json')

def osmFISH():
    return Experiment.from_json('https://dmf0bdeheu4zf.cloudfront.net/20180919/osmFISH/experiment.json')
