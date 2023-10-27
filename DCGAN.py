# Import
import tensorflow
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import uuid
import os
import numpy as np

# Initialize variables
NUM_EPOCHS = 50
BUFFER_SIZE = 30000
BATCH_SIZE = 28
NOISE_DIMENSION = 75
UNIQUE_RUN_ID = str(uuid.uuid4())
PRINT_STATS_AFTER_BATCH = 50
OPTIMIZER_LR = 0.0002
OPTIMIZER_BETAS = (0.5, 0.999)
WEIGHT_INIT_STDDEV = 0.02

# Initialize loss function, init schema and optimizers
cross_entropy_loss = tensorflow.keras.losses.BinaryCrossentropy(from_logits=True)
weight_init = tensorflow.keras.initializers.RandomNormal(stddev=WEIGHT_INIT_STDDEV)
generator_optimizer = tensorflow.keras.optimizers.Adam(OPTIMIZER_LR, \
  beta_1=OPTIMIZER_BETAS[0], beta_2=OPTIMIZER_BETAS[1])
discriminator_optimizer = tensorflow.keras.optimizers.Adam(OPTIMIZER_LR, \
  beta_1=OPTIMIZER_BETAS[0], beta_2=OPTIMIZER_BETAS[1])


def make_directory_for_run():
  """ Make a directory for this training run. """
  print(f'Preparing training run {UNIQUE_RUN_ID}')
  if not os.path.exists('./runs'):
    os.mkdir('./runs')
  os.mkdir(f'./runs/{UNIQUE_RUN_ID}')


def generate_image(generator, epoch = 0, batch = 0):
  """ Generate subplots with generated examples. """
  images = []
  noise = generate_noise(BATCH_SIZE)
  images = generator(noise, training=False)
  plt.figure(figsize=(10, 10))
  for i in range(16):
    # Get image and reshape
    image = images[i]
    image = np.reshape(image, (28, 28))
    # Plot
    plt.subplot(4, 4, i+1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
  if not os.path.exists(f'./runs/{UNIQUE_RUN_ID}/images'):
    os.mkdir(f'./runs/{UNIQUE_RUN_ID}/images')
  plt.savefig(f'./runs/{UNIQUE_RUN_ID}/images/epoch{epoch}_batch{batch}.jpg')


def load_data():
  """ Load data """
  (images, _), (_, _) = tensorflow.keras.datasets.mnist.load_data()
  images = images.reshape(images.shape[0], 28, 28, 1)
  images = images.astype('float32')
  images = (images - 127.5) / 127.5
  return tensorflow.data.Dataset.from_tensor_slices(images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def create_generator():
  """ Create Generator """
  generator = tensorflow.keras.Sequential()
  # Input block
  generator.add(layers.Dense(7*7*128, use_bias=False, input_shape=(NOISE_DIMENSION,), \
    kernel_initializer=weight_init))
  generator.add(layers.BatchNormalization())
  generator.add(layers.LeakyReLU())
  # Reshape 1D Tensor into 3D
  generator.add(layers.Reshape((7, 7, 128)))
  # First upsampling block
  generator.add(layers.Conv2DTranspose(56, (5, 5), strides=(1, 1), padding='same', use_bias=False, \
    kernel_initializer=weight_init))
  generator.add(layers.BatchNormalization())
  generator.add(layers.LeakyReLU())
  # Second upsampling block
  generator.add(layers.Conv2DTranspose(28, (5, 5), strides=(2, 2), padding='same', use_bias=False, \
    kernel_initializer=weight_init))
  generator.add(layers.BatchNormalization())
  generator.add(layers.LeakyReLU())
  # Third upsampling block: note tanh, specific for DCGAN
  generator.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh', \
    kernel_initializer=weight_init))
  # Return generator
  return generator


def generate_noise(number_of_images = 1, noise_dimension = NOISE_DIMENSION):
  """ Generate noise for number_of_images images, with a specific noise_dimension """
  return tensorflow.random.normal([number_of_images, noise_dimension])
  

def create_discriminator():
  """ Create Discriminator """
  discriminator = tensorflow.keras.Sequential()
  # First Convolutional block
  discriminator.add(layers.Conv2D(28, (5, 5), strides=(2, 2), padding='same',
                                    input_shape=[28, 28, 1], kernel_initializer=weight_init))
  discriminator.add(layers.LeakyReLU())
  discriminator.add(layers.Dropout(0.5))
  # Second Convolutional block
  discriminator.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=weight_init))
  discriminator.add(layers.LeakyReLU())
  discriminator.add(layers.Dropout(0.5))
  # Flatten and generate output prediction
  discriminator.add(layers.Flatten())
  discriminator.add(layers.Dense(1, kernel_initializer=weight_init, activation='sigmoid'))
  # Return discriminator
  return discriminator


def compute_generator_loss(predicted_fake):
  """ Compute cross entropy loss for the generator """
  return cross_entropy_loss(tensorflow.ones_like(predicted_fake), predicted_fake)


def compute_discriminator_loss(predicted_real, predicted_fake):
  """ Compute discriminator loss """
  loss_on_reals = cross_entropy_loss(tensorflow.ones_like(predicted_real), predicted_real)
  loss_on_fakes = cross_entropy_loss(tensorflow.zeros_like(predicted_fake), predicted_fake)
  return loss_on_reals + loss_on_fakes


def save_models(generator, discriminator, epoch):
  """ Save models at specific point in time. """
  tensorflow.keras.models.save_model(
    generator,
    f'./runs/{UNIQUE_RUN_ID}/generator_{epoch}.model',
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
  )
  tensorflow.keras.models.save_model(
    discriminator,
    f'./runs/{UNIQUE_RUN_ID}/discriminator{epoch}.model',
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
  )
  

def print_training_progress(batch, generator_loss, discriminator_loss):
  """ Print training progress. """
  print('Losses after mini-batch %5d: generator %e, discriminator %e' %
        (batch, generator_loss, discriminator_loss))


@tensorflow.function
def perform_train_step(real_images, generator, discriminator):
  """ Perform one training step with Gradient Tapes """
  # Generate noise
  noise = generate_noise(BATCH_SIZE)
  # Feed forward and loss computation for one batch
  with tensorflow.GradientTape() as discriminator_tape, \
      tensorflow.GradientTape() as generator_tape:
        # Generate images
        generated_images = generator(noise, training=True)
        # Discriminate generated and real images
        discriminated_generated_images = discriminator(generated_images, training=True)
        discriminated_real_images = discriminator(real_images, training=True)
        # Compute loss
        generator_loss = compute_generator_loss(discriminated_generated_images)
        discriminator_loss = compute_discriminator_loss(discriminated_real_images, discriminated_generated_images)
  # Compute gradients
  generator_gradients = generator_tape.gradient(generator_loss, generator.trainable_variables)
  discriminator_gradients = discriminator_tape.gradient(discriminator_loss, discriminator.trainable_variables)
  # Optimize model using gradients
  generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
  # Return generator and discriminator losses
  return (generator_loss, discriminator_loss)
        
  
def train_gan(num_epochs, image_data, generator, discriminator):
  """ Train the GAN """
  # Perform one training step per batch for every epoch
  for epoch_no in range(num_epochs):
    num_batches = image_data.__len__()
    print(f'Starting epoch {epoch_no+1} with {num_batches} batches...')
    batch_no = 0
    # Iterate over batches within epoch
    for batch in image_data:
      generator_loss, discriminator_loss = perform_train_step(batch, generator, discriminator)
      batch_no += 1
      # Print statistics and generate image after every n-th batch
      if batch_no % PRINT_STATS_AFTER_BATCH == 0:
        print_training_progress(batch_no, generator_loss, discriminator_loss)
        generate_image(generator, epoch_no, batch_no)
    # Save models on epoch completion.
    save_models(generator, discriminator, epoch_no)
  # Finished :-)
  print(f'Finished unique run {UNIQUE_RUN_ID}')


def run_gan():
  """ Initialization and training """
  # Make run directory
  make_directory_for_run()
  # Set random seed
  tensorflow.random.set_seed(42)
  # Get image data
  data = load_data()
  # Create generator and discriminator
  generator = create_generator()
  discriminator = create_discriminator()
  # Train the GAN
  print('Training GAN ...')
  train_gan(NUM_EPOCHS, data, generator, discriminator)
  

if __name__ == '__main__':
  run_gan()


  
