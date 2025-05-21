import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Data & model configuration
img_width, img_height = x_train.shape[1], x_train.shape[2]
batch_size = 128
no_epochs = 100
validation_split = 0.2
verbosity = 1
latent_dim = 2
num_channels = 1

# Reshape data
x_train = x_train.reshape(x_train.shape[0], img_height, img_width, num_channels)
x_test = x_test.reshape(x_test.shape[0], img_height, img_width, num_channels)

# Parse numbers as floats
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize data
x_train = x_train / 255
x_test = x_test / 255

# Create the VAE as a custom Model subclass
class VAE(keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.img_width = img_width
        self.img_height = img_height
        
        # Create encoder
        self.encoder = self._build_encoder()
        
        # Create decoder
        self.decoder = self._build_decoder()
        
        # Metrics
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        
    def _build_encoder(self):
        encoder_inputs = layers.Input(shape=(img_height, img_width, num_channels))
        x = layers.Conv2D(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(encoder_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(20, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Save shape of the last feature map for the decoder
        self.shape_before_flattening = tuple(x.shape)[1:-1]
        
        z_mean = layers.Dense(latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
        
        # Sample z
        z = layers.Lambda(self.sampling, name='z')([z_mean, z_log_var])
        
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
        return encoder
    
    def _build_decoder(self):
        decoder_inputs = layers.Input(shape=(latent_dim,))
        
        # Calculate dimensions for reshaping based on encoder architecture
        # For MNIST with 28x28 input, after two stride-2 convolutions, dimensions are 7x7
        decoder_dense_shape = 7 * 7 * 16  # This matches our encoder's output before flattening
        
        x = layers.Dense(decoder_dense_shape, activation='relu')(decoder_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Reshape((7, 7, 16))(x)  # Reshape to match encoder output before flattening
        
        x = layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        decoder_outputs = layers.Conv2DTranspose(filters=num_channels, kernel_size=3, activation='sigmoid', padding='same')(x)
        
        decoder = keras.Model(decoder_inputs, decoder_outputs, name='decoder')
        return decoder
    
    def sampling(self, args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def call(self, inputs):
        # Forward pass through the model
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def train_step(self, data):
        # In train_step, data can be either a tensor or a tuple of two tensors
        # If it's a tuple, we're expecting data and targets, but with VAE we don't need targets
        # so we just take the first element
        if isinstance(data, tuple):
            data = data[0]
            
        with tf.GradientTape() as tape:
            # Forward pass
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # Calculate reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2)
                )
            ) * self.img_width * self.img_height
            
            # Calculate KL divergence loss
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=1
                )
            )
            
            # Total loss
            total_loss = reconstruction_loss + kl_loss
        
        # Compute gradients and update weights
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def test_step(self, data):
        # In test_step, data can be either a tensor or a tuple of two tensors
        # If it's a tuple, we're expecting data and targets, but with VAE we don't need targets
        # so we just take the first element
        if isinstance(data, tuple):
            data = data[0]
            
        # Forward pass
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        
        # Calculate reconstruction loss
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(data, reconstruction),
                axis=(1, 2)
            )
        ) * self.img_width * self.img_height
        
        # Calculate KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1
            )
        )
        
        # Total loss
        total_loss = reconstruction_loss + kl_loss
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# Create the VAE model
vae = VAE(latent_dim=latent_dim)

# Compile the model (the loss is handled in the train_step method)
vae.compile(optimizer=keras.optimizers.Adam())

# Display model summary
vae.encoder.summary()
vae.decoder.summary()

# Train the model - FIXED: use x_test as validation data without repeating it
vae.fit(x_train, epochs=no_epochs, batch_size=batch_size, validation_data=(x_test, None))

# Visualization functions
def plot_latent_space(vae, n=30, figsize=15):
    # Display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = vae.encoder.predict(x_test)
    plt.figure(figsize=(figsize, figsize))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()

def plot_latent_images(vae, n=15, digit_size=28):
    # Display a n*n 2D manifold of digits
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n, num_channels))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size, num_channels)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    
    # matplotlib.pyplot.imshow() needs a 2D array, or a 3D array with the third dimension being of shape 3 or 4!
    # So reshape if necessary
    fig_shape = np.shape(figure)
    if fig_shape[2] == 1:
        figure = figure.reshape((fig_shape[0], fig_shape[1]))
    
    plt.imshow(figure)
    plt.show()

# Generate visualizations
plot_latent_space(vae)
plot_latent_images(vae)

