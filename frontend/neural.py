import os
import tensorflow as tf
import tensorflow_hub as hub
from django.conf import settings
from io import StringIO

# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

print(tf.version.VERSION)

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

import numpy as np
import PIL.Image
import time
import functools

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)

  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)


# plt.subplot(1, 2, 1)
# imshow(content_image, 'Content Image')

# plt.subplot(1, 2, 2)
# imshow(style_image, 'Style Image')

# print("Style", stylized_image)

def get_neural_style(style_path):

  content_path =  '/Users/gautham/Documents/Documents - gBookPro/Berkeley MIMS/CalHacks/prototype/Artify/static/images/images/content.jpg'
  style_path = '/Users/gautham/Documents/Documents - gBookPro/Berkeley MIMS/CalHacks/prototype/Artify/static/images/images/' + style_path + '.jpeg'

  # content_path = '/Users/gautham/Documents/Documents - gBookPro/Berkeley MIMS/CalHacks/prototype/Artify/static/images/images/content.jpg'
  # style_path = '/Users/gautham/Documents/Documents - gBookPro/Berkeley MIMS/CalHacks/prototype/Artify/static/images/images/chill_1.jpeg'

  content_image = load_img(content_path)
  style_image = load_img(style_path)

  hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
  stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

  # tensor_to_image(stylized_image).save("./static/images/Sample_final.png")

  return tensor_to_image(stylized_image)




