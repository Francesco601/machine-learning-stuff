import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


(x_train, _), (_, _) = keras.datasets.mnist.load_data() # Load the MNIST dataset

x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0 # Normalize and reshape the images

def make_generator_model():
    model = keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


def make_discriminator_model():
    model = keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


def create_gan(generator, discriminator):
    discriminator.trainable = False

    gan_input = keras.Input(shape=(100,))
    gan_output = discriminator(generator(gan_input))
    gan = keras.Model(gan_input, gan_output)

    return gan

def train_gan(gan, generator, discriminator, epochs, batch_size):
    for epoch in range(epochs):
        for batch in range(x_train.shape[0] // batch_size):
            # Step 1: Train the discriminator
            noise = np.random.normal(0, 1, (batch_size, 100))
            generated_images = generator.predict(noise)

            real_images = x_train[batch * batch_size : (batch + 1) * batch_size]

            with tf.GradientTape() as disc_tape:
                real_output = discriminator(real_images, training=True)
                fake_output = discriminator(generated_images, training=True)
                disc_loss = discriminator_loss(real_output, fake_output)

            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

            # Step 2: Train the generator
            noise = np.random.normal(0, 1, (batch_size, 100))
            with tf.GradientTape() as gen_tape:
                generated_images = generator(noise, training=True)
                fake_output = discriminator(generated_images, training=True)
                gen_loss = generator_loss(fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

            if batch % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch}/{x_train.shape[0] // batch_size}, Generator Loss: {gen_loss}, Discriminator Loss: {disc_loss}")

    return generator



def generate_images(generator):
    noise = np.random.normal(0, 1, (25, 100))
    generated_images = generator.predict(noise)

    fig, axes = plt.subplots(5, 5, figsize=(10,10))
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated_images[i, :, :, 0], cmap='gray')
        ax.axis('off')

    plt.show()

generator = make_generator_model()
discriminator = make_discriminator_model()
gan = create_gan(generator, discriminator)
generator = train_gan(gan, generator, discriminator, epochs=20, batch_size=256)
generate_images(generator)



