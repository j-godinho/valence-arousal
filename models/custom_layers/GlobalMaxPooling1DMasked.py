from keras import backend as K
from keras.engine import InputSpec
from keras.engine.topology import Layer
import numpy as np

from keras.layers.pooling import  GlobalMaxPooling1D

class GlobalMaxPooling1DMasked(GlobalMaxPooling1D):
    """
    This pooling layer accepts the temporal sequence output by a recurrent layer
    and performs temporal pooling, looking at only the non-masked portion of the sequence.
    The pooling layer converts the entire variable-length hidden vector sequence
    into a single hidden vector.
    Modified from https://github.com/fchollet/keras/issues/2151 so code also
    works on tensorflow backend. Updated syntax to match Keras 2.0 spec.
    Args:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        3D tensor with shape: `(samples, steps, features)`.
        input shape: (nb_samples, nb_timesteps, nb_features)
        output shape: (nb_samples, nb_features)
    Examples:
        > x = Bidirectional(GRU(128, return_sequences=True))(x)
        > x = TemporalMaxPooling()(x)
    """
    def __init__(self, **kwargs):
        super(GlobalMaxPooling1DMasked, self).__init__(**kwargs)
        self.supports_masking = True
        self.input_spec = InputSpec(ndim=3)

    def call(self, x, mask=None):
        return (super().call(x))

    def compute_mask(self, input, mask):
        # do not pass the mask to the next layers
        return None