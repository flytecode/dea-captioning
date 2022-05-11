from numpy import expand_dims
import tensorflow as tf
from attention import MultiHeadedAttention
from utils import PositionWiseFeedForward

class EncoderLayer():

    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        
        self.multi_attention = MultiHeadedAttention(d_model, d_k, d_v, h, dropout)
        self.position_feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

    def __call__(self, queries, keys, values, mask=None, attention_weights=None):

        attention = self.multi_attention(queries, keys, values, mask, attention_weights)
        return self.position_feed_forward(attention)

class MultiLevelEncoder(tf.keras.Model):

    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,):
        self.d_model = d_model
        self.dropout = dropout
        self.padding_idx = padding_idx

        self.mult_layers = [
            EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout)
            for _ in range(N)]
    
    def forward(self, input, attention_weights=None):
        attention_mask = expand_dims(
            expand_dims(
            tf.reduce_sum(input, -1) == self.padding_idx, 
            axis=1), 
            axis=1)

        outs = []
        out = input
        for l in self.mult_layers:
            out = l(out, out, out, attention_mask, attention_weights)
            outs.append(expand_dims(out, 1))

        outs = tf.concat(outs, 1)
        return outs, attention_mask

class MemoryAugmentedEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super(MemoryAugmentedEncoder, self).__init__(N, padding_idx)
        self.dense = tf.keras.layers.Dense(self.d_model, activation='relu')
        self.dropout = tf.keras.layers.Dropout(self.dropout)
        self.norm  = tf.keras.layers.LayerNormalization()
    

    def __call__(self, input, attention_weights=None):
        dense = self.dense(input)
        dropout = self.dropout(dense)
        norm = self.norm(dropout)
        return super(MemoryAugmentedEncoder, self).forward(norm, attention_weights)

