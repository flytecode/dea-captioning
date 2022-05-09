import tensorflow as tf
import copy
import numpy as np
from models.containers import ModuleList

class Transformer():
    def __init__(self, bos_idx, encoder, decoder):
        super(Transformer, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.decoder = decoder
        self.register_state("enc_output", None)
        self.register_state("mask_enc", None)
        
    @property
    def d_model(self):
        return self.decoder.d_model

    def forward(self, images, seq, *args):
        enc_output, mask_enc = self.encoder(images)
        dec_output = self.decoder(seq, enc_output, mask_enc)
        return dec_output
    
    def init_state(self, b_s, device):
        return [tf.zeros(((b_s,0))), None, None]
    
    def step(self, t, prev_output, visual, seq, **kwargs):
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

class TransformerEnsemble():
    def __init__(self, model: Transformer, weight_files):
        super(TransformerEnsemble, self).__init__()
        self.n = len(weight_files)
        self.models = ModuleList([copy.deepcopy(model) for _ in range(self.n)])
        for i in range(self.n):
            state_dict_i = np.load(weight_files[i])["state_dict"]
            self.models[i].load_state_dict(state_dict_i)

    def step(self, t, prev_output, visual, seq, **kwargs):
        out_ensemble = []
        for i in range(self.n):
            out_i = self.models[i].step(t, prev_output, visual, seq, **kwargs)
            out_ensemble.append(tf.exapnd_dims(out_i, 0))

        return np.mean(tf.concat(out_ensemble, 0), dim=0)
