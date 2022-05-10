import tensorflow as tf
import numpy as np



def positional_embedding(range, model_dim):
    range = range.view(-1, 1)
    dim = np.arange(model_dim // 2, dtype=tf.float32).view(1, -1)
    sin = tf.math.sin(input / 10000 ** (2 * dim / model_dim))
    cos = tf.math.cos(input / 10000 ** (2 * dim / model_dim))

    out = tf.zeros(input.shape[0], model_dim)
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out

def sinusoidal_encoding(len, model_dim, padding_idx=None):
    range = tf.arange(len, dtype=tf.float32)
    encoding = positional_embedding(range, model_dim)

    return encoding


class PositionWiseFeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model=512, d_ff=2048, dropout=.1, identity_map_reordering=False):
        super(PositionWiseFeedForward, self).__init__()
        self.identity_map_reordering = identity_map_reordering

        # CURRENT PROGRESS
        self.fc1 = tf.keras.layers.Dense()
        self.fc2 = tf.nn.Linear(d_ff, d_model)
        self.dropout = tf.nn.Dropout(p=dropout)
        self.dropout_2 = tf.nn.Dropout(p=dropout)
        self.layer_norm = tf.nn.LayerNorm(d_model)
    def forward(self, input):
        if self.identity_map_reordering:
            out = self.layer_norm(input)
            out = self.fc2(self.dropout_2(F.relu(self.fc1(out))))
            out = input + self.dropout(tf.relu(out))
        else:
            out = self.fc2(self.dropout_2(F.relu(self.fc1(input))))
            out = self.dropout(out)
            out = self.layer_norm(input + out)
        return out