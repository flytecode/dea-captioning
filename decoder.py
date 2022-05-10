from matplotlib.cbook import sanitize_sequence
import tensorflow as tf
import numpy as np


from attention import MultiHeadedAttention, MemoryAttention
from utils import SinEncoding, sinusoidal_encoding, PositionWiseFeedForward


class MeshedDecoderLayer(tf.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048,
                    dropout=.1, self_att_module=None,
                    enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(MeshedDecoderLayer, self).__init__()
        self.self_att = MultiHeadedAttention(d_model, d_k, d_v, h, dropout, 
                            
                            attention_module_kwargs=self_att_module_kwargs)
        self.enc_att = MultiHeadedAttention(d_model, d_k, d_v, h, dropout,
                                          
                                          attention_module_kwargs=enc_att_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.fc_alpha1 = tf.keras.layers.Dense(d_model)
        self.fc_alpha2 = tf.keras.layers.Dense(d_model)
        self.fc_alpha3 = tf.keras.layers.Dense(d_model)

    def __call__(self, input, enc_output, mask_pad, mask_self_att, mask_enc_att):
        self_att = self.self_att(input, input, input, mask_self_att)
        mask_pad = tf.cast(mask_pad, dtype=tf.float32)
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

        forward = self.pwff(enc_attention)
        forward *= mask_pad

        return forward

class MeshedDecoder(tf.keras.Model):
    def __init__(self, vocab_size, max_len, N_dec, padding_index, d_model=512, d_k=64,
                    d_v=64, h=8, d_ff=2048, dropout=.1,
                 self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
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