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

  initializer = tf.random_normal_initializer(0, 0.2)

  last = Conv2DTranspose(filters = 3,
                         kernel_size = 4,
                         strides = 2,
                         padding = "same",
                         kernel_initializer = initializer,
                         activation = "tanh")
  x = inputs
  s = []

  concat = Concatenate()

  for down in down_stack:
    x = down(x)
    s = s.append(x)
  
  s =  reversed(s[:-1])

  for up in up_stack:
    x = up(x)
    x = concat([x, sk])

  last = last(x)

  return last(x)

generator = Generator()
gen_output = generator(((inimg+1)*255), training = False)
plt.imshow(gen_output[0,...])


def Discriminator():
  ini = Input(shape=[None, None, 3], name="input_img")
  gen = Input(shape=[None, None, 3], name="gener_img")

  con = concatenate([ini, gen])

  initializer = tf.random_normal_initializer(0, 0.2)
 
  down1 = decomposer(64, apply_batchnorm=False)(con)
  down2 = decomposer(128)
  down3 = decomposer(256)
  down4 = decomposer(512)

  last = tf.keras.layers.Conv2D(filters=1,
                                kernle_size=4,
                                strides=1,
                                kernel_initializer=initializer,
                                padding="same")(down4)

  return tf.keras.Model(inputs=[ini, gen], outputs=last)

discriminator = Discriminator()

disc_out = discriminator()

def discriminator_loss(disc_real_output, disc_generated_output):

  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss
  
  import os

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_prefix = os.path.join(CKPATH, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer = discriminator_optimizer,
                                 generator=generator,
                                 disciminator=discriminator)
def generate_images(model, test_input, tar, save_filename=False, display_imgs=True):
  prediction = model(test_input, training=True)
  if save_filename:
    tf.keras.preprocessing.image.save_img(PATH + 'output/' + save:filename + '.jpg', prediction[0,...])

  plt.figure(figsize=(10,10))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  if display_imgs:
    for i in range(3):
      plt.subplot(1, 3, i+1)
      plt.title(title[i])
      plt.imshow(display_list[i] * 0.5 + 0.5)
      plt.axis('off')
      
  plt.show()
  
  def train_step(input_image, terget):

  with GradientTape() as gen_tape, GradientTape() as discr_tape:
    ouput_image = generator(input_image, training=True)

    output_gen_discr = discriminator([ouptut_image, input_image], training=True)

    output_trg_discr = discriminator([target, input_image], training=True)

    discr_loss = descriminator_loss()

    generator_grads = gen_tape.gradients(gen_loss, generator.trainable_variables)

    discriminator_grads = discr_tape.gradients(discr_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients()
    
    from IPython.display import clear.output

def train(dataset, epochs):
  for epoch in range(epochs):

    imgi = 0
    for input_image, target in dataset:
      print('epoch ' + str(epoch) + ' - train' + str(imgi)+'/'+str(len(tr_urls)))
      imgi+=1
      train_step(input_image, target)

    clear_output(wait=True)

    for inp, tar in test:dataset.take(5):
      generate_images(generator, inp, tar, str(imgi) + '_' + str(epoch, display_imgs=True))

    if (epoch + 1) %25==0:
      checkpoint.save(file_prefix = checkpoint_prefix)
