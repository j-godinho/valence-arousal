from keras import backend as K
from keras.engine import InputSpec
from keras.engine.topology import Layer
import numpy as np
from keras import initializers

class Attention(Layer):
    def __init__(self, **kwargs):
        self.attention = None
        self.init = initializers.get('normal')
        self.supports_masking = True
        super(Attention, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='kernel', shape=(input_shape[-1],), initializer='normal', trainable=True)
        super(Attention, self).build(input_shape)
    def call(self, x, mask=None):
        #eij = K.tanh(K.dot(x, self.W))
        eij = K.tanh(K.squeeze(K.dot(x, K.expand_dims(self.W)), axis=-1))
        ai = K.exp(eij)
        #weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')
        weights = ai/K.expand_dims(K.sum(ai, axis=1), 1)
        #weighted_input = x*weights.dimshuffle(0,1,'x')
        weighted_input = x*K.expand_dims(weights,2)
        self.attention = weights
        return K.sum(weighted_input, axis=1)
        #return weights
    def compute_mask(self, input, mask):
        # do not pass the mask to the next layers
        return None
    def get_output_shape_for(self, input_shape): return (input_shape[0], input_shape[-1])
    #def get_output_shape_for(self, input_shape): return (input_shape[0], input_shape[-2], 1)
    def compute_output_shape(self, input_shape): return self.get_output_shape_for(input_shape)