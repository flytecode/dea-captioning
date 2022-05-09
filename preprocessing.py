from pycocotools.coco import COCO as pyCOCO
import numpy as np
import tensorflow as tf
import string
import PIL
import itertools
from collections import Counter

class CocoLoader:
    
    def __init__(self, annotations, id_file):
        self.annotations = annotations
        self.dataset = pyCOCO(self.annotations)
        self.id_file = id_file
        self.ids = np.load(self.id_file)
        np.random.shuffle(self.ids)
        self.curr_batch = 0
        self.create_dict()

    def create_dict(self):
        ann_ids = self.dataset.getAnnIds(imgIds=self.ids)
        full_sentences = [banger["caption"].lower().translate(
            str.maketrans('', '', string.punctuation)).split() 
            for banger in self.dataset.loadAnns(ann_ids)]

        all_words = list(itertools.chain.from_iterable(full_sentences))
        items = list(Counter(all_words).items())
        items.sort(reverse=True, key= lambda x: x[1])

        words_accounted = 0
        self.vocab_dict = {"<STOP>":0, "<UNK>":1}

        curr_val = 0
        while words_accounted < len(all_words) * .95:
            self.vocab_dict[items[curr_val][0]] = curr_val + 2
            words_accounted += items[curr_val][1]
            curr_val += 1

        self.vocab_size = len(self.vocab_dict)
        self.dict_vocab = {integer:word for word, integer in self.vocab_dict.items()}

    def get_vocab_dicts(self):
        return self.vocab_dict, self.dict_vocab

    def load(self, batches):

        ann_ids = []
        k = 0
        while len(ann_ids) < batches:
            ann_ids += self.dataset.getAnnIds(imgIds=[self.ids[k]])
            k += 1
        ann_ids = ann_ids[:batches]

        self.curr_batch += batches

        full_sentences = [banger["caption"].lower().translate(
            str.maketrans('', '', string.punctuation)).split() 
            for banger in self.dataset.loadAnns(ann_ids)]
        
        for j, sentence in enumerate(full_sentences):
            if len(sentence) < 15:
                while len(full_sentences[j]) < 15:
                    full_sentences[j] += ["<STOP>"]
            if len(sentence) > 15:
                full_sentences[j] = sentence[:15]

            for i, word in enumerate(full_sentences[j]):
                if word not in self.vocab_dict:
                    full_sentences[j][i] = self.vocab_dict["<UNK>"]
                else:
                    full_sentences[j][i] = self.vocab_dict[word]
    

        return tf.one_hot(np.array(full_sentences), self.vocab_size)

loader = CocoLoader("annotations/captions_train2014.json", "annotations/coco_train_ids.npy")
print(loader.vocab_dict)
loader.load(256)
max_h, max_w = 0, 0
