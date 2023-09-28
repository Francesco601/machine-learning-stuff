import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def build_generator(latent_dim):
    model = keras.Sequential()
    model.add(layers.Dense(256, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(784, activation='tanh'))
    model.add(layers.Reshape((28, 28, 1)))
    return model

def build_discriminator():
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28, 1)))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = keras.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

(X_train, _), (_, _) = keras.datasets.mnist.load_data()
X_train = X_train.reshape((-1, 28, 28, 1)).astype("float32") / 255.0

latent_dim = 100
epochs = 50
batch_size = 128

generator = build_generator(latent_dim)
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

gan.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002), loss='binary_crossentropy')

for epoch in range(epochs):
    for step in range(X_train.shape[0] // batch_size):
        # Generate random noise as input for the generator
        noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
        
        # Generate fake images using the generator
        fake_images = generator.predict(noise)
        
        # Select a random batch of real images
        real_images = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]
        
        # Create training batches for the discriminator
        x = np.concatenate((real_images, fake_images))
        y = np.array([1] * batch_size + [0] * batch_size)
        
        # Train the discriminator
        discriminator_loss = discriminator.train_on_batch(x, y)
        
        # Generate new random noise
        noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
        
        # Defining the desired output for the generator (tricking the discriminator)
        y_gen = np.array([1] * batch_size)
        
        # Train the generator
        gan_loss = gan.train_on_batch(noise, y_gen)
    
    print(f"Epoch: {epoch} | Discriminator loss: {discriminator_loss:.4f} | GAN loss: {gan_loss:.4f}")


n_images = 10
noise = np.random.normal(0, 1, size=(n_images, latent_dim))
generated_images = generator.predict(noise)

# Plotting the generated images
fig, axs = plt.subplots(1, n_images, figsize=(10, 2))
for i in range(n_images):
    axs[i].imshow(generated_images[i, :, :, 0], cmap='gray')
    axs[i].axis('off')
plt.tight_layout()
plt.show()

