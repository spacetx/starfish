.. _decoding:

Decoding
========

Decoders can be imported using ``starfish.spots.Decoder``, which registers all classes that subclass
``DecoderAlgorithmBase``:

.. code-block:: python

    from starfish.spots import Decoder

.. contents::

Per Round Max Channel Decoder
-----------------------------

.. autoclass:: starfish.spots._decoder.per_round_max_channel_decoder.PerRoundMaxChannelDecoder
    :members:
