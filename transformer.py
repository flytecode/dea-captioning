import tensorflow as tf
import copy
import numpy as np


class Transformer(tf.keras.Model):
   """
   Transformer model implementing both the encdoer and decoder subarchitectures.
   """
   def __init__(self, bos_idx, encoder, decoder):
       """
       Initializes the model after being passed in necessary subarchitectures.
 
       Params:
       bos_idx -- Index of padding word added to the start of words during
                   training.
       encoder -- already initialized encoder subarchitecture.
       decoder -- already initialized decoder subarchitecture.
       """
 
       super(Transformer, self).__init__()
       self.bos_idx = bos_idx
       self.encoder = encoder
       self.decoder = decoder
 
      
   @property
   def d_model(self):
       """
       Dimensions of the model.
       """
       return self.decoder.d_model
 
   def __call__(self, images, seq, *args):
       """
       Calls forward pass on encoder and decoder to get output.
 
       Params:
       images -- images to be processed
       seq    -- ground truth sentence values
 
       Returns: Result of forward pass
       """
       enc_output, mask_enc = self.encoder(images)
       dec_output = self.decoder(seq, enc_output, mask_enc)
       return dec_output
  
   def init_state(self, b_s, device):
       """Returns initial state"""
       return [tf.zeros(((b_s,0))), None, None]
  
   def step(self, t, prev_output, visual, seq, **kwargs):
       """
       Completes the forward pass for a single step the model.
 
       Params:
       prev_output -- Last output of the model in a sentence
       seq         -- ground truth sentence
       visual      -- image being analyzed
 
       Returns: output of that single step.
       """
       it = None
       if t == 0:
           self.enc_output, self.mask_enc = self.encoder(visual)
           if isinstance(visual, tf.Tensor):
               it = visual.data.new_full((visual.shape[0], 1), self.bos_idx)
           else:
               it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
       else:
           it = prev_output
      
       return self.decoder(it, self.enc_output, self.mask_enc)

