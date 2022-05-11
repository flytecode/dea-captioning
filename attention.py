import tensorflow as tf 
import numpy as np
import math
import random

import tensorflow as tf 
import numpy as np
import math
import random

class MemoryAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, d_k, d_v, h, m):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param m: Number of Memory Slots
        '''
        super(MemoryAttention, self).__init__()
        print("How the fuck did I get here")
        self.queryDense = tf.keras.layers.Dense(h * d_k, name='gigglePuss')
        self.keysDense = tf.keras.layers.Dense(h * d_k)
        self.valuesDense = tf.keras.layers.Dense(h * d_v)
        self.outDense = tf.keras.layers.Dense(d_model)
        print("I am in here ma")
        self.memoryKeys = tf.Variable(
            tf.random_normal_initializer(0,1/d_k)(shape=[1,m,h*d_k]), name="THis is a var")
        self.memoryVals = tf.Variable(
            tf.random_normal_initializer(0, 1/m)(shape=[1,m, h*d_v]))

        self.model_size = d_model
        self.kq_size = d_k
        self.value_size = d_v
        self.heads = h
        self.memories = m



    def __call__(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        :params queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:'''
        
        b_s, num_queries = queries.shape[:2]
        num_keys = keys.shape[1]

        memory_keys = np.sqrt(self.kq_size)*np.reshape(self.memoryKeys,
            [1, self.memories, self.heads*self.kq_size])
        memory_values = np.sqrt(self.memories)*np.reshape(self.memoryVals,
            [1, self.memories, self.heads*self.value_size])

        queries_l = tf.transpose(tf.reshape(self.queryDense(queries),
            [b_s, num_queries, self.heads, self.kq_size]), perm=[0,2,1,3])

        keys_l = tf.transpose(tf.reshape(tf.concat(
            [self.keysDense(keys), tf.broadcast_to(memory_keys, [b_s, self.memories, self.heads*self.kq_size])],1),
            [b_s, num_keys+self.memories, self.heads, self.kq_size]), perm=[0,2,3,1])

        vals_l = tf.transpose(tf.reshape(tf.concat(
            [self.valuesDense(values),  tf.broadcast_to(memory_keys, [b_s, self.memories, self.heads*self.value_size])],1),
            [b_s, num_keys+self.memories, self.heads, self.value_size]), perm=[0,2,1,3])
        
        attention = tf.matmul(queries_l, keys_l)/ np.sqrt(self.kq_size)
        if attention_weights is not None:
            attention = tf.concat([
                attention[:,:,:,:num_keys]*attention_weights, 
                attention[:,:,:,num_keys:]], -1)
        if attention_mask is not None:
            pass
            #attention[:,:,:, :num_keys].assign(tf.boolean_mask(
             #   attention[:,:,:,:num_keys], tf.broadcast_to(attention_mask, attention[:,:,:,:num_keys].shape)))
                
        attention = tf.nn.softmax(attention)
        output = tf.reshape(
            tf.transpose(tf.matmul(attention, vals_l), perm= [0,2,1,3]),
            [b_s, num_queries, self.heads*self.value_size])
        output - self.outDense(output)

        return output

class MultiHeadedAttention(tf.keras.Model):
    '''
    This acts as the multiheaded attention layer with Dropout and normalization
    '''

    def __init__(self, d_model, d_k, d_v, h, m, dropout=.1, identity_map_reordering = False,
                     attention_module_kwargs=None):
        super(MultiHeadedAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.queryDense = tf.keras.layers.Dense(h * d_k)
        self.keysDense = tf.keras.layers.Dense(h * d_k)
        self.valuesDense = tf.keras.layers.Dense(h * d_v)
        self.outDense = tf.keras.layers.Dense(d_model)
        keysInit = tf.random_normal_initializer(0,1/d_k)

        self.memoryKeys = tf.Variable(
            keysInit(shape=[1,m,h*d_k], dtype=tf.float32), name="THis is a var", trainable=True)

        self.memoryVals = tf.Variable(
            tf.random_normal_initializer(0, 1/m)(shape=[1,m, tf.cast(h*d_v, dtype=tf.int32)], 
            dtype=tf.float32), trainable=True)

        self.model_size = d_model
        self.kq_size = d_k
        self.value_size = d_v
        self.heads = h
        self.memories = m

        self.dropout = tf.keras.layers.Dropout(dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def __call__(self, queries, keys, values, attention_mask=None, attention_weights=None):
        
        b_s, num_queries = queries.shape[:2]
        num_keys = keys.shape[1]

        memory_keys = np.sqrt(self.kq_size)*np.reshape(self.memoryKeys,
            [1, self.memories, self.heads*self.kq_size])
        memory_values = np.sqrt(self.memories)*np.reshape(self.memoryVals,
            [1, self.memories, self.heads*self.value_size])

        queries_l = tf.transpose(tf.reshape(self.queryDense(queries),
            [b_s, num_queries, self.heads, self.kq_size]), perm=[0,2,1,3])

        keys_l = tf.transpose(tf.reshape(tf.concat(
            [self.keysDense(keys), tf.broadcast_to(memory_keys, [b_s, self.memories, self.heads*self.kq_size])],1),
            [b_s, num_keys+self.memories, self.heads, self.kq_size]), perm=[0,2,3,1])

        vals_l = tf.transpose(tf.reshape(tf.concat(
            [self.valuesDense(values),  tf.broadcast_to(memory_values, [b_s, self.memories, self.heads*self.value_size])],1),
            [b_s, num_keys+self.memories, self.heads, self.value_size]), perm=[0,2,1,3])
        
        attention = tf.matmul(queries_l, keys_l)/ np.sqrt(self.kq_size)
        if attention_weights is not None:
            attention = tf.concat([
                attention[:,:,:,:num_keys]*attention_weights, 
                attention[:,:,:,num_keys:]], -1)
        if attention_mask is not None:
            pass
            #attention[:,:,:, :num_keys].assign(tf.boolean_mask(
             #   attention[:,:,:,:num_keys], tf.broadcast_to(attention_mask, attention[:,:,:,:num_keys].shape)))
                
        attention = tf.nn.softmax(attention)
        output = tf.reshape(
            tf.transpose(tf.matmul(attention, vals_l), perm= [0,2,1,3]),
            [b_s, num_queries, self.heads*self.value_size])
        output = self.outDense(output)
        output = self.dropout(output)
        output = self.layer_norm(queries+output)
        return output