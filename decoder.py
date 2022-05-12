from matplotlib.cbook import sanitize_sequence
import tensorflow as tf
import numpy as np


from attention import MultiHeadedAttention
from utils import SinEncoding


class MeshedDecoderLayer(tf.keras.layers.Layer):
   """
   Class for a single layer in the decoder. Uses a number of dense layers,
   along with self attention and encoder attention, to create an output.
   Notably, both implementations of attention use MultiHeadedAttention, thus
   implementing the meshed memory system.
   """
   def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048,
                   dropout=.1, self_att_module_kwargs=None, enc_att_module_kwargs=None):
       """
       Initializes the layer and all sub-architectures:
 
       Params:
       d_model                 -- dimensionality of the model
       d_k                     -- key dimension size
       d_v                     -- value dimensions
       h                       -- number of heads in attention layers
       d_ff                    -- number of features
       dropout                 -- droupout rate
       self_att_module_kwargs  -- kwargs for self attention if necessary
       enc_att_module_kwargs   -- kwargs for encoder attention if necessary
       """
       super(MeshedDecoderLayer, self).__init__()
       self.self_att = MultiHeadedAttention(d_model, d_k, d_v, h, 40, dropout,
                          
                           attention_module_kwargs=self_att_module_kwargs)
       self.enc_att = MultiHeadedAttention(d_model, d_k, d_v, h, 40, dropout,
                                        
                                         attention_module_kwargs=enc_att_module_kwargs)
       self.dense1 = tf.keras.layers.Dense(d_ff, name="FirstDenseDEcode")
       self.dense2 = tf.keras.layers.Dense(d_model, name="SecondDenseDecode")
       self.dropout1 = tf.keras.layers.Dropout(dropout)
       self.dropout2 = tf.keras.layers.Dropout(dropout)
       self.layer_normalize = tf.keras.layers.LayerNormalization(name="NormalizeHereDecode")
 
 
       self.fc_alpha1 = tf.keras.layers.Dense(d_model, name="Dense in Decoder")
       self.fc_alpha2 = tf.keras.layers.Dense(d_model)
       self.fc_alpha3 = tf.keras.layers.Dense(d_model)
 
   def __call__(self, input, enc_output, mask_pad, mask_self_att, mask_enc_att):
       """
       Applies forward pass to the model, passing through attention layers and
       final dense layers for each attention head.
 
       Params:
       input           -- input to the model
       enc_output      -- output of the encoder beforehand
       mask_pad        -- mask for padding terms
       mask_self_att   -- mask restricting in self attention
       mask_enc_att    -- mask restricting in encoder attention
       """
       self_att = self.self_att(input, input, input, mask_self_att)
       mask_pad = tf.cast(mask_pad, dtype=tf.float32)
       mask_enc_att = tf.cast(mask_enc_att, dtype=tf.float32)
       self_att *= mask_pad
 
       enc_att1 = self.enc_att(self_att, enc_output[:, 0], enc_output[:,0],
                       mask_enc_att*mask_pad)
       enc_att2 = self.enc_att(self_att, enc_output[:, 1], enc_output[:,0],
                       mask_enc_att*mask_pad)
       enc_att3 = self.enc_att(self_att, enc_output[:, 2], enc_output[:,0],
                       mask_enc_att*mask_pad)
 
       alphaA = tf.nn.sigmoid(self.fc_alpha1(tf.concat([self_att, enc_att1], -1)))
       alphaB = tf.nn.sigmoid(self.fc_alpha2(tf.concat([self_att, enc_att2], -1)))
       alphaC = tf.nn.sigmoid(self.fc_alpha3(tf.concat([self_att, enc_att3], -1)))
 
       enc_attention = (enc_att1*alphaA+enc_att2*alphaB+enc_att3*alphaC)/np.sqrt(3)
       enc_attention *= mask_pad
 
       out = self.dense2(self.dropout2(tf.nn.relu(self.dense1(enc_attention))))
       out = self.dropout1(out)
       out = self.layer_normalize(enc_attention+out)
       out *= mask_pad
 
       return out
 
class MeshedDecoder(tf.keras.Model):
   """
   Full architecture of the encoder, using a number of smaller encoder layers as defined above.
   Also includes normalization and dense layers to apply to outputs.
   """   
   def __init__(self, vocab_size, max_len, N_dec, padding_index, d_model=512, d_k=64,
                   d_v=64, h=8, d_ff=2048, dropout=.1,
                self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
       """
       Initializes the decoder and all internal layers.       
 
       Params:
       vocab_size  -- number of words in the vocabulary
       max_len     -- maximum length of a sentence
       N_dec       -- number of decoder layers implemented
       padding_idx -- the index in sentence arrays corresponding to padding
       d_in        -- dimensionality of the input
       d_model     -- dimensions of the model
       d_k         -- dimension of keys
       d_v         -- dimension of vals
       h           -- height
       d_ff        -- dimension of features
       dropout     -- dropout rate
       """
       super(MeshedDecoder, self).__init__()
       self.d_model = d_model
       self.word_embeddings = tf.keras.layers.Embedding(vocab_size, d_model)
       self.position_embedding = tf.keras.layers.Embedding(vocab_size, d_model,
                       embeddings_initializer=SinEncoding)
       self.multi_layers = [MeshedDecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module,
                               enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
                               enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)]
       self.dense = tf.keras.layers.Dense(vocab_size)
       self.max_length = max_len
       self.padding_index = padding_index
       self.number = N_dec
 
   def __call__(self, input, encoder_output, mask_encoder):
       """
       Calls a forward pass on the decpder and all internal decoder layers.
 
       Params:
       input             -- Inputs
       attention_weights -- weights for the pass through multiheaded attention, if any.
       encoder_output    -- output of the encoder
       mask_encoder      -- mask from encoder
 
       Returns:
       outs -- probability distribution among words
       """
       b_s, seq_len = input.shape[:2]
       mask_queries = tf.expand_dims((input != self.padding_index),-1)
 
       mask_self_attention = tf.linalg.band_part(tf.ones((seq_len,seq_len)),0,-1)
 
       mask_self_attention = tf.expand_dims(tf.expand_dims(mask_self_attention,0),0)
       mask_self_attention += tf.cast(tf.expand_dims(tf.expand_dims((input==self.padding_index),1),1), dtype=tf.float32)
       mask_self_attention = tf.math.greater(mask_self_attention, 0)
       seq = tf.reshape(np.arange(1, seq_len+1), [1,-1])
       seq = tf.broadcast_to(seq, [b_s, seq.shape[1]])
       #mask = tf.squeeze(mask_queries, -1)==0
       #seq = tf.boolean_mask(seq, mask)
       out = self.word_embeddings(input) + self.position_embedding(seq)
       for i, l in enumerate(self.multi_layers):
           out = l(out, encoder_output, mask_queries, mask_self_attention, mask_encoder)
 
       out = self.dense(out)
       return tf.nn.log_softmax(out, axis=-1)