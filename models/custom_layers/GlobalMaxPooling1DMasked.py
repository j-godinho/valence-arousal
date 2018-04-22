from keras import backend as K
from keras.engine import InputSpec
from keras.engine.topology import Layer
import numpy as np

from keras.layers.pooling import GlobalMaxPooling1D

class GlobalMaxPooling1DMasked(GlobalMaxPooling1D):

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(GlobalMaxPooling1D, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        if mask is None: mask = K.sum(K.ones_like(x), axis=-1)
        mask = mask.dimshuffle(0, 1, "x")
        masked_data = K.switch(K.equal(mask, 0), K.constant(-np.inf), x)
        return masked_data.max(axis=1)