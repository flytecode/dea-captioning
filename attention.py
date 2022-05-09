import tensorflow as tf 
import numpy as np
import math
import random

class memoryAttention(tf.Module):

    def __init__(self, d_model, d_k, d_v, h, m):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param m: Number of Memory Slots
        '''
        super(memoryAttention, self).__init__()

        self.queryDense = tf.keras.layers.Dense(h * d_k)
        self.keysDense = tf.keras.layers.Dense(h * d_k)
        self.valuesDense = tf.keras.layers.Dense(h * d_v)
        self.outDense = tf.keras.layers.Dense(d_model)

        self.memoryKeys = tf.Variable(
            tf.random_normal_initializer(0,1/d_k)(size=[1,m,h*d_k]))
        self.memoryVals = tf.Variable(
            tf.random_normal_initializer(0, 1/m)(size=[1,m, h*d_v]))

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

        memory_keys = np.sqrt(self.kq_size)*self.memoryKeys.expand(
            b_s, self.memories, self.heads*self.kq_size)
        memory_values = np.sqrt(self.memories)*self.memoryVals.expand(
            b_s, self.memories, self.heads*self.value_size)

        queries_l = tf.transpose(self.queryDense(queries).reshape(
            b_s, num_queries, self.heads, self.kq_size), perm=[0,2,1,3])
        keys_l = tf.transpose(tf.concat(
            [self.keysDense(keys), memory_keys],1).reshape(
            b_s, num_keys+self.memories, self.heads, self.kq_size), perm=[0,2,3,1])
        vals_l = tf.transpose(tf.concat(
            [self.valsDense(values), memory_values],1).reshape(
            b_s, num_keys+self.memories, self.heads, self.value_size), perm=[0,2,1,3])
        
        attention = tf.matmul(queries_l, keys_l)/ np.sqrt(self.kq_size)
        if attention_weights is not None:
            attention = tf.concat([
                attention[:,:,:,:num_keys]*attention_weights, 
                attention[:,:,:,num_keys:]], -1)
        if attention_mask is not None:
            attention[:,:,:, :num_keys] = tf.boolean_mask(
                attention[:,:,:,:num_keys], attention_mask)
        attention = tf.nn.softmax(attention)
        output = tf.reshape(
            tf.transpose(tf.matmul(attention, vals_l), perm= [0,2,1,3]),
            [b_s, num_queries, self.heads*self.value_size])
        output - self.outDense(output)

        return output

class MultiHeadedAttention(Module):
    '''
    This acts as the multiheaded attention layer with Dropout and normalization
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering = False,
                    attention_module_kwargs=None):
        super(MultiHeadedAttention, self).__init__()

        self.identity_map_reordering = identity_map_reordering
        self.attention = memoryAttention(d_model = d_model, d_k=d_k, d_v=d_v, h=h)

        self.dropout = tf.keras.layers.Dropout(dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def __call__(self, queries, keys, values, attention_mask=None, attention_weights=None):
        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)

            out = self.attention(q_norm, k_norm, v_norm, attention_mask, attention_weights)
            out = queries + self.dropout(tf.nn.relu(out))
        else:
            out = self.attention(queries, keys, values, attention_mask, attention_weights)
            out = self.dropout(out)
            out = self.layer_norm(queries+out)
        return out