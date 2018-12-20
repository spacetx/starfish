class AlgorithmBase:
    """
    This is the base class of any algorithm that starfish exposes.

    Subclasses of this base class are paired with subclasses of PipelineComponent. The subclasses of
    PipelineComponent retrieve subclasses of the paired AlgorithmBase. Together, the two classes
    enable starfish to expose a paired API and CLI.

    Examples
    --------

    PipelineComponent: `starfish.image._segmentation.Segmentation(PipelineComponent)`

    AlgorithmBase: `starfish.image._segmentation._base.SegmentationAlgorithmBase(AlgorithmBase)`

    Implementing Algorithms:
    - `starfish.image._segmentation.watershed.Watershed(SegmentationAlgorithmBase)`

    This pattern exposes the API as follows:

    `starfish.image.Segmentation.<implementing algorithm (Watershed)>`

    and the CLI as:

    `$> starfish segmentation watershed`

    To create an entirely new group of related algorithms, like `Segmentation`, a new subclass of
    both `AlgorithmBase` and `PipelineComponent` must be created.

    To add to an existing group of algorithms like "Segmentation", an algorithm implementation must
    subclass the corresponding subclass of `AlgorithmBase`. In this case,
    `SegmentationAlgorithmBase`.

    See Also
    --------
    starfish.pipeline.pipelinecomponent.py
    """
    @classmethod
    def _get_algorithm_name(cls):
        """
        Returns the name of the algorithm.  This should be a valid python identifier, i.e.,
        https://docs.python.org/3/reference/lexical_analysis.html#identifiers
        """
        return cls.__name__
