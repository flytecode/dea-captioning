import tensorflow as tf
import numpy as np

def positional_embedding(range, model_dim):
    range = np.reshape(range, [-1, 1])
    dim = np.reshape(np.arange(model_dim // 2, dtype=np.float32), [1,-1])
    sin = tf.math.sin(range / 10000 ** (2 * dim / model_dim))
    cos = tf.math.cos(range / 10000 ** (2 * dim / model_dim))

    out = np.zeros([range.shape[0], model_dim])
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out

class SinEncoding(tf.keras.initializers.Initializer):
    def __init__(self):
        super().__init__()
    def __call__(self, shape, dtype=None, **kwargs):
        max_len, model_dim = shape[0],shape[1]
        range = np.arange(max_len, dtype= np.float32)
        encoding = positional_embedding(range, model_dim)
        encoding[0] = 0
        return encoding

def sinusoidal_encoding(len, model_dim, padding_idx=None):
    range = np.arange(len, dtype= np.float32)
    encoding = positional_embedding(range, model_dim)
    return encoding
