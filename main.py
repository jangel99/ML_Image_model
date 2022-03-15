import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

PATH = "/content/drive/MyDrive/Uni/TFG"

#definimos las rutas de los datos
Input_PATH = PATH + '/InputAnimals'
Output_PATH = PATH + '/TargetAnimals'
Checkpoint_PATH = PATH + 'Checkpoints'

img_data = !ls -1 "{Input_PATH}"

print(len(img_data))

#Organizar datos de entrenamiento
n = 500

train_n = round(n * 0.80)

randurls = np.copy(img_data)

np.random.seed()
np.random.shuffle(randurls)

tr_urls = randurls[:train_n]
ts_urls = randurls[train_n:n]

print(len(img_data)), len(tr_urls), len(ts_urls) 
#definimos tamaño de la imagen
IMG_WIDTH = 256
IMG_HEIGHT = 256

def resize(input_img, target_image, height, width):

  input_img = tf.image.resize(input_img, [height, width])
  target_img = tf.image.resize(target_img, [height, width])

  return input_img, target_img

def normalize (input_img,target_img):
  input_img = (input_img / 127.5) -1
  target_img = (input_img / 127.5) -1
  return input_img, target_img


def random_jitter(input_img,target_img):

  input_img, target_img = resize(input_img, target_img, 286, 286)

  stacked_img = tf.stack([input_img, target_img], axis=0)
  cropped_img = tf.image.random_crop(stacked_img, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  input_img, target_img = cropped_img[0], cropped_img[1]

  if tf.random.uniform(()) > 0.5:

    input_img = tf.image.flip_left_right(input_img)
    target_img = tf.image.flip_left_right(target_img)

  return input_img, target_img

def load_img (filename, augment=True):

  input_img = tf.cast(tf.image.decode_jpeg(tf.io.read_file(Input_PATH + '/' + filename)), tf.float32)[..., :3]
  target_img = tf.cast(tf.image.decode_jpeg(tf.io.read_file(Output_PATH + '/' + filename)), tf.float32)[..., :3]

  input_img, re_img = resize(input_img, re_img, IMG_HEIGHT, IMG_WIDTH)

  if augment:
    input_img, re_img = random_jitter(input_img, re_img)

  imput_img, re_img = normalize(input_img, re_img)

  return input_img, re_img


def load_train_image(filename):
  return load_img(filename, True)

def load_test_image(filename):
  return load_img(filename, False)

plt.imshow(load_train_image(randurls[0]))

train_dataset = tf.data.Dataset.from_tensor_slices(tr_urls)
train_dataset = train_dataset.map(load_train_image, num_parallel_calls = tf.data.experimental.AUTOTUNE)
train_dateset = train_dataset.batch(1)

test_dataset = tf.data.Dataset.from_tensor_slices(tr_urls)
test_dataset = test_dataset.map(load_test_image, num_parallel_calls = tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(1)

from tensorflow.keras.layers import *

def decomposer(filters):

  result = Sequential()
  initializer = tf.random_normal_initializer(0, 0.02)
  result.add(Conv2D(filters,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=initializer,
                    use_bias=not apply_batchnorm))
  if apply_batchnorm:
    result.add(BatchNormalization())

  result.add(leakyRelu())

  return result

decomposer(64)

def upscale(filters, apply_dropout = True):
  reult = Sequential()

  initializer = tf.random_normal_initializer(0, 0.02)

  result.add(Conv2D(filters,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=initializer,
                    use_bias=False))

  result.add(BatchNormalization())

  if apply_dropout:
    result.add(Dropout(0.5))

  result.add(Relu())

  return result

  upscale(64)
  
  def Generator():
  inputs = tf.keras.layers.Input(shape=[None,None,3])

  down_stack = [
    decomposer(64, apply_batchnorm=False), 
    decomposer(128),
    decomposer(256),
    decomposer(512),
    decomposer(512),
    decomposer(512),
    decomposer(512),
    decomposer(512),
  ]

  up_stack = [
    upscale(512, apply_dropout=True),
    upscale(512, apply_dropout=True),
    upscale(512, apply_dropout=True),
    upscale(512),
    upscale(256),
    upscale(128),
    upscale(64),
  ]

  last = Conv2DTranspose(filters = 3,
                         kernel_size = 4,
                         strides = 2,
                         padding = "same",
                         kernel_initializer = initializer,
                         activation = "tanh")
  x = inputs

  for down in down_stack:
    x = down(x)

  for up in up_stack:
    x = up(x)

  return last(x)
