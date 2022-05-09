import tensorflow as tf 

import matplotlib.pyplot as plt
import collections
import random
import numpy as np
import os, time, json
from PIL import Image
from tqdm import tqdm

BATCH_SIZE = 24


annotation_folder = '/annotations/'
if not os.path.exists(os.path.abspath('.') + annotation_folder):
  annotation_zip = tf.keras.utils.get_file('captions.zip',
                                           cache_subdir=os.path.abspath('.'),
                                           origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                           extract=True)
  annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2014.json'
  os.remove(annotation_zip)

image_folder = "/train2014/"
if not os.path.exists(os.path.abspath(".")+image_folder):
    image_zip = tf.keras.utils.get_file("train2014.zip",
                                        cache_subdir= os.path.abspath("."),
                                        origin = "http://images.cocodataset.org/zips/train2014.zip",
                                        extract=True)
    path = os.path.dirname(image_zip)+image_folder
    os.remove(image_zip)
else:
    path = os.path.abspath(".")+image_folder


annotation_file = os.path.abspath(".")+"/annotations/captions_train2014.json"
with open(annotation_file, "r") as annot_file:
    annotations = json.load(annot_file)

image_path_to_caption = collections.defaultdict(list)
for value in annotations["annotations"]:
    caption = f"<start> {value['caption']} <end>"
    image_path = path+"COCO_train2014_"+"%012d.jpg" % (value["image_id"])
    image_path_to_caption[image_path].append(caption)

image_path = list(image_path_to_caption.keys())
random.shuffle(image_path)


###CHANGE 6000 TO TRAIN ON MORE IMAGES

train_image_paths = image_path[:6000]

training_captions = []
img_name_vector = []

for image_path in train_image_paths:
    caption_list = image_path_to_caption[image_path]
    training_captions.extend(caption_list)
    img_name_vector.extend([image_path]*len(caption_list))


print(training_captions[0:5])
im = Image.open(img_name_vector[0])

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.keras.layers.Resizing(299,299)(image)
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image, image_path

image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights="imagenet")
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

encoding_train = sorted(set(img_name_vector))

image_dataset = tf.data.Dataset.from_tensor_slices(encoding_train)
image_dataset = image_dataset.map(load_image, 
                    num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE)

for image, path in image_dataset:
    batch_features = image_features_extract_model(image)
    batch_features = tf.reshape(batch_features,
                        (batch_features.shape[0], -1, batch_features.shape[3]))

    for batch_feature, path in zip(batch_features, path):
        path_of_feature = 

