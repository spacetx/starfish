class AlgorithmBase:
    """
    This is the base class of any algorithm that starfish exposes. This base class is retrieved by
    the PipelineComponent class to register the Algorithm to the CLI and API.

    New algorithm classes, like "segmentation" must subclass AlgorithmBase. See, e.g.
    starfish.image._segmentation._base.SegmentationAlgorithmBase

    All algorithm implementations, like "Watershed", must then subclass their AlgorithmBase. In this
    case, SegmentationAlgorithmBase. See, e.g.
    starfish.image._segmentation.watershed.Watershed

    This pattern combines with a PipelineComponent, in this case named "Segmentation" to enable
    any implementing methods to be accessed from the API as:

    starfish.image.Segmentation.<implementing algorithm (Watershed)>

    and from the CLI as:

    $> starfish segmentation watershed
    """
    @classmethod
    def _get_algorithm_name(cls):
        """
        Returns the name of the algorithm.  This should be a valid python identifier, i.e.,
        https://docs.python.org/3/reference/lexical_analysis.html#identifiers
        """
        return cls.__name__
