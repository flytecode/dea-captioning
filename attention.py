import tensorflow as tf 
import numpy as np
import math
import random

class dotProductAttention():

    def __init__(self, d_model, d_k, d_v, h):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''

        self.queryDense = tf.keras.layers.Dense(h * d_k)
        self.keysDense = tf.keras.layers.Dense(h * d_k)
        self.valuesDense = tf.keras.layers.Dense(h * d_v)

        self.outDense = tf.keras.layers.Dense(d_model)

        self.model_size = d_model
        
        
class multiHeadedAttention():
    pass