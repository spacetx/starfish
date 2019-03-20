from starfish.pipeline.pipelinecomponent import PipelineComponentType
from starfish.spots import Decoder


def test_pipelinecomponent_by_name():
    assert PipelineComponentType.get_pipeline_component_type_by_name("decode") == Decoder
