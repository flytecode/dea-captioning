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

for image, path in tqdm(image_dataset):
    batch_features = image_features_extract_model(image)
    batch_features = tf.reshape(batch_features,
                        (batch_features.shape[0], -1, batch_features.shape[3]))

    for batch_feature, path in zip(batch_features, path):
        path_of_feature = path.numpy().decode("utf-8")
        np.save(path_of_feature, batch_feature.numpy())

caption_dataset = tf.data.Dataset.from_tensor_slices(training_captions)

def standardize(inputs):
    inputs = tf.strings.lower(inputs)
    return tf.strings.regex_replace(inputs, 
                                r"!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~", "")


max_length = 25
vocab_size = 6000
tokenizer = tf.keras.layers.TextVectorization(max_tokens = vocab_size,
                                                standardize=standardize,
                                                output_sequence_length=max_length)
tokenizer.adapt(caption_dataset)

caption_vector = caption_dataset.map(lambda x: tokenizer(x))

word_to_index = tf.keras.layers.StringLookup(mask_token="", 
                vocabulary= tokenizer.get_vocabulary())

index_to_word = tf.keras.layers.StringLookup(mask_token="", 
                vocabulary= tokenizer.get_vocabulary(),
                invert=True)

image_to_caption_vector = collections.defaultdict(list)
for image, caption in zip(img_name_vector, caption_vector):
    image_to_caption_vector[image].append(caption)

image_keys = list(image_to_caption_vector.keys())
random.shuffle(image_keys)

slice_index = int(len(image_keys)*.8)
image_name_train_keys = image_keys[:slice_index] 
image_name_validation_keys = image_keys[slice_index:]

image_name_train = []

caption_train = []

for train_image in image_name_train_keys:
    caption_length = len(image_to_caption_vector[train_image])
    image_name_train.extend([train_image]*caption_length)
    caption_train.extend(image_to_caption_vector[train_image])

image_name_validation = []
caption_validation = []
for validation_image in image_name_validation_keys:
    caption_length = len(image_to_caption_vector[validation_image])
    image_name_validation.extend([validation_image]*caption_length)
    caption_validation.extend([validation_image])

print(len(image_name_train), len(caption_train), len(image_name_validation), len(caption_validation))



BATCH_SIZE=64

BUFFER_SIZE = 1000
embedding_dimensions = 128
units = 512
iterations = len(image_name_train)//BATCH_SIZE

features_shape = 2048
attention_features_shape = 64

def map_function(image_name, caption):
    image_tensor = np.load(image_name.decode("utf-8")+".npy")
    return image_tensor, caption
training_dataset = tf.data.Dataset.from_tensor_slices((image_name_train, caption_train))
training_dataset = training_dataset.map(lambda item1, item2: tf.numpy_function(
    map_function, [item1, item2], [tf.float32, tf.int64]),
    num_parallel_calls=tf.data.AUTOTUNE)

training_dataset = training_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

training_dataset = training_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)



def train():
    pass


if __name__ == "__main__":
    train()