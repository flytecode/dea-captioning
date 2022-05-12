import tensorflow as tf 
import numpy as np
import math
import random

import tensorflow as tf 
import numpy as np
import math
import random

class MultiHeadedAttention(tf.keras.Model):
   '''
   This acts as the multiheaded attention layer with Dropout and normalization.
   Also includes the "Meshed Memory Attention" system unique to this
   implementation, concatenating into the model an extra variable for the keys
   to allow each encoder/decoder layer to remember relationships.
   '''
 
   def __init__(self, d_model, d_k, d_v, h, m, dropout=.1, identity_map_reordering = False,
                    attention_module_kwargs=None):
       """
       Initializes all internal layers, along with the memory keys and values.
 
       Params:
       d_model                 -- dimensions of the model
       d_k                     -- dimension of keys
       d_v                     -- dimension of values
       h                       -- number heads
       m                       -- number of memory heads
       dropout                 -- dropout rate
       identity_map_reordering -- whether or not to reorder inputs
       """
       super(MultiHeadedAttention, self).__init__()
       self.identity_map_reordering = identity_map_reordering
       self.queryDense = tf.keras.layers.Dense(h * d_k)
       self.keysDense = tf.keras.layers.Dense(h * d_k)
       self.valuesDense = tf.keras.layers.Dense(h * d_v)
       self.outDense = tf.keras.layers.Dense(d_model)
       keysInit = tf.random_normal_initializer(0,1/d_k)
 
       self.memoryKeys = tf.Variable(
           keysInit(shape=[1,m,h*d_k], dtype=tf.float32), name="MultiHead Memory Keys", trainable=True)
 
       self.memoryVals = tf.Variable(
           tf.random_normal_initializer(0, 1/m)(shape=[1,m, h*d_v],
           dtype=tf.float32), name="MultiHead Memory Vals", trainable=True)
       self.model_size = d_model
       self.kq_size = d_k
       self.value_size = d_v
       self.heads = h
       self.memories = m
 
       self.dropout = tf.keras.layers.Dropout(dropout)
       self.layer_norm = tf.keras.layers.LayerNormalization()
 
   def __call__(self, queries, keys, values, attention_mask=None, attention_weights=None):
       """
       Forward pass on the multiheaded attention.
 
       Params:
       queries             -- query tensor
       values              -- value tensor
       keys                -- keys tensor
       attention_mask      -- Mask for attention values, if there is one
       attention_weights   -- Weights for attention, if there are any
 
       Returns: output of forward pass
       """
      
       b_s, num_queries = queries.shape[:2]
       num_keys = keys.shape[1]
 
       memory_keys = np.sqrt(self.kq_size)*self.memoryKeys
       memory_values = np.sqrt(self.memories)*self.memoryVals
 
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
