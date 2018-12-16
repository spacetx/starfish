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
    """Return osmFISH data from Codeluppi et al. 2018

    This function returns a single round of a single field of view from the 16 field of view study
    if use_test_data is True, or three fields of view containing all rounds of data if use_test_data
    is False.

    Parameters
    ----------
    use_test_data : bool
        If True, return one round from one field of view, suitable for testing (default False)

    Notes
    -----
    - osmFISH fields of view are quite large (14, 2, 45, 2048, 2048) which takes up approximately
      21 gb in memory. Use the non-test data with care.

    See Also
    --------
    Codeluppi et al. 2018: https://www.nature.com/articles/s41592-018-0175-z
    """
    if use_test_data:
        return Experiment.from_json(
            'https://d2nhj9g34unfro.cloudfront.net/20181005/osmFISH/experiment.json')
    return Experiment.from_json(
        "https://d2nhj9g34unfro.cloudfront.net/20181031/osmFISH/experiment.json")


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


def ImagingMassCytometry(use_test_data: bool=False):
    return Experiment.from_json(
        "https://d2nhj9g34unfro.cloudfront.net/browse/formatted/20181023/"
        "imaging_cytof/BodenmillerBreastCancerSamples/experiment.json"
    )


def SeqFISH(use_test_data: bool=False):
    """Loads a SeqFISH field of view generated from cultured mES cells.

    Parameters
    ----------
    use_test_data : bool
        If true, return a small region of testing data that was visually determined to contain two
        cells.

    Notes
    -----
    SeqFISH fields of view are quite large (12, 5, 29, 2048, 2048) and take up approximately
    5 gb in memory. Use the non-test data with care.

    See Also
    --------
    Manuscript for Intron-SeqFISH: https://doi.org/10.1016/j.cell.2018.05.035

    """
    suffix = "-TEST" if use_test_data else ""
    url = (
        f"https://d2nhj9g34unfro.cloudfront.net/browse/formatted/20181211/"
        f"seqfish{suffix}/experiment.json"
    )
    return Experiment.from_json(url)
