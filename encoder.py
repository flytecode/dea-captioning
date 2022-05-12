
import tensorflow as tf
from attention import MultiHeadedAttention

import tensorflow as tf
from attention import MultiHeadedAttention
from utils import PositionWiseFeedForward
 
class EncoderLayer(tf.Module):
   """
   Basic encoder layer. Implements Multiheaded attention along with
   dense layers.
   """
 
   def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
       """
       Initializes all layers inside and the instance of the MultiHeadedAttention
       function.
 
       Params:
       d_model -- dimensions of the model
       d_k     -- dimension of keys
       d_v     -- dimension of vals
       h       -- height
       d_ff    -- dimension of features
       dropout -- dropout rate
       """
       super(EncoderLayer, self).__init__()
       self.multi_attention = MultiHeadedAttention(d_model, d_k, d_v, h, 40, dropout)
 
       self.dense1 = tf.keras.layers.Dense(d_ff, name="FirstDense")
       self.dense2 = tf.keras.layers.Dense(d_model, name="SecondDense")
       self.dropout1 = tf.keras.layers.Dropout(dropout)
       self.dropout2 = tf.keras.layers.Dropout(dropout)
       self.layer_normalize = tf.keras.layers.LayerNormalization(name="NormalizeHere")
 
   def __call__(self, queries, keys, values, mask=None, attention_weights=None):
       """
       Completes a forward pass of the model first through the multiheaded attention,
       then the dense layer.
 
       Params:
       queries           -- query tensor
       keys              -- key tensor
       values            -- value tensor
       mask              -- Attention mask
       attention_weights -- weights for pass through attention layer
 
       Returns: Output of the forward pass.
       """
 
       attention = self.multi_attention(queries, keys, values, mask, attention_weights)
      
       out = self.dense2(self.dropout2(tf.nn.relu(self.dense1(attention))))
       out = self.dropout1(out)
       out = self.layer_normalize(attention+out)
       return out
 
class MemoryAugmentedEncoder(tf.keras.Model):
   """
   Full architecture of the encoder, using a number of smaller encoder layers as defined above.
   Also includes normalization and dense layers to apply to outputs.
   """   
   def __init__(self, N, padding_idx, d_in=2048, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
       """
       Initializes the encoder and all internal layers.       
 
       Params:
       N           -- number of encoder layers implemented
       padding_idx -- the index in sentence arrays corresponding to padding
       d_in        -- dimensionality of the input
       d_model     -- dimensions of the model
       d_k         -- dimension of keys
       d_v         -- dimension of vals
       h           -- height
       d_ff        -- dimension of features
       dropout     -- dropout rate
       """
       super(MemoryAugmentedEncoder, self).__init__()
       self.d_model = d_model
       self.dense = tf.keras.layers.Dense(self.d_model, activation='relu', name="ThisDense")
       self.dropout = tf.keras.layers.Dropout(dropout, name="Encoder Dropout")
       self.norm  = tf.keras.layers.LayerNormalization(name="LayerNormInEncoder")
 
 
      
       self.padding_idx = padding_idx
 
       self.mult_layers = [
           EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout)
           for i in range(N)]
 
   def __call__(self, input, attention_weights=None):
       """
       Calls a forward pass on the encoder and all internal encoder layers.
 
       Params:
       input             -- Inputs
       attention_weights -- weights for the pass through multiheaded attention, if any.
 
       Returns:
       outs           -- Outputs of each attention layer
       attention_mask -- Mask to apply to attention
       """
 
       dense = self.dense(input)
       dropout = self.dropout(dense)
       norm = self.norm(dropout)
       attention_mask = tf.expand_dims(
           tf.expand_dims(
           tf.reduce_sum(norm, -1) == self.padding_idx,
           axis=1),
           axis=1)
 
       outs = []
       out = norm
       for l in self.mult_layers:
           out = l(out, out, out, attention_mask, attention_weights)
           outs.append(tf.expand_dims(out, 1))
 
       outs = tf.concat(outs, 1)
       return outs, attention_mask
