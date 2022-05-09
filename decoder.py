import tensorflow as transformer
import numpy as np


from attention import dotProductAttention
from utils import sinusoid_encoding_table, PositionWiseFeedForward


class MeshedDecoderLayer(Module):
    def __init__(self)
