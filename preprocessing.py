from pycocotools.coco import COCO as pyCOCO
import numpy as np
import tensorflow as tf
import string
import itertools
from collections import Counter

class CocoLoader:
    
    def __init__(self, annotations, id_file):
        self.annotations = annotations
        self.dataset = pyCOCO(self.annotations)
        self.id_file = id_file
        self.ids = np.load(self.id_file)
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
        print(items)

        words_accounted = 0
        self.vocab_dict = {"<STOP>":0, "<UNK>":1}

        curr_val = 0
        while words_accounted < len(all_words) * .9:
            print(words_accounted)
            self.vocab_dict[curr_val + 2] = items[curr_val][0]
            words_accounted += items[curr_val][1]
            curr_val += 1

        self.vocab_size = len(self.vocab_dict)


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

            for i, word in enumerate(sentence):
                if word not in self.vocab_dict:
                    sentence[i] = self.vocab_dict["<UNK>"]
                else:
                    sentence[i] = self.vocab_dict[word]
        
        img_ids = [banger["image_id"] for banger in self.dataset.loadAnns(ann_ids)]
        images = self.dataset.loadImgs(img_ids)
        print(images)

        return tf.one_hot(np.array(full_sentences), self.vocab_size)

loader = CocoLoader("annotations/captions_train2014.json", "annotations/coco_train_ids.npy")
print(loader.load(10))
