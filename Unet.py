import os
import tensorflow
from tensorflow.keras.layers import Conv2D,\
	MaxPool2D, Conv2DTranspose, Input, Activation,\
	Concatenate, CenterCrop
from tensorflow.keras import Model
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.optimizers import schedules, Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


'''
	U-NET CONFIGURATION
'''
def configuration():
	''' Get configuration. '''

	return dict(
		data_train_prc = 80,
		data_val_prc = 90,
		data_test_prc = 100,
		num_filters_start = 64,
		num_unet_blocks = 3,
		num_filters_end = 3,
		input_width = 100,
		input_height = 100,
		mask_width = 60,
		mask_height = 60,
		input_dim = 3,
		optimizer = Adam,
		loss = SparseCategoricalCrossentropy,
		initializer = HeNormal(),
		batch_size = 50,
		buffer_size = 50,
		num_epochs = 25,
		metrics = ['accuracy'],
		dataset_path = os.path.join(os.getcwd(), 'data'),
		class_weights = tensorflow.constant([1.0, 1.0, 2.0]),
		validation_sub_splits = 5,
		lr_schedule_percentages = [0.2, 0.5, 0.8],
		lr_schedule_values = [3e-4, 1e-4, 1e-5, 1e-6],
		lr_schedule_class = schedules.PiecewiseConstantDecay
	)


'''
	U-NET BUILDING BLOCKS
'''

def conv_block(x, filters, last_block):
	'''
		U-Net convolutional block.
		Used for downsampling in the contracting path.
	'''
	config = configuration()

	# First Conv segment
	x = Conv2D(filters, (3, 3),\
		kernel_initializer=config.get("initializer"))(x)
	x = Activation("relu")(x)

	# Second Conv segment
	x = Conv2D(filters, (3, 3),\
		kernel_initializer=config.get("initializer"))(x)
	x = Activation("relu")(x)

	# Keep Conv output for skip input
	skip_input = x

	# Apply pooling if not last block
	if not last_block:
		x = MaxPool2D((2, 2), strides=(2,2))(x)

	return x, skip_input


def contracting_path(x):
	'''
		U-Net contracting path.
		Initializes multiple convolutional blocks for
		downsampling.
	'''
	config = configuration()

	# Compute the number of feature map filters per block
	num_filters = [compute_number_of_filters(index)\
			for index in range(config.get("num_unet_blocks"))]

	# Create container for the skip input Tensors
	skip_inputs = []

	# Pass input x through all convolutional blocks and
	# add skip input Tensor to skip_inputs if not last block
	for index, block_num_filters in enumerate(num_filters):

		last_block = index == len(num_filters)-1
		x, skip_input = conv_block(x, block_num_filters,\
			last_block)

		if not last_block:
			skip_inputs.append(skip_input)

	return x, skip_inputs


def upconv_block(x, filters, skip_input, last_block = False):
	'''
		U-Net upsampling block.
		Used for upsampling in the expansive path.
	'''
	config = configuration()

	# Perform upsampling
	x = Conv2DTranspose(filters//2, (2, 2), strides=(2, 2),\
		kernel_initializer=config.get("initializer"))(x)
	shp = x.shape

	# Crop the skip input, keep the center
	cropped_skip_input = CenterCrop(height = x.shape[1],\
		width = x.shape[2])(skip_input)

	# Concatenate skip input with x
	concat_input = Concatenate(axis=-1)([cropped_skip_input, x])

	# First Conv segment
	x = Conv2D(filters//2, (3, 3),
		kernel_initializer=config.get("initializer"))(concat_input)
	x = Activation("relu")(x)

	# Second Conv segment
	x = Conv2D(filters//2, (3, 3),
		kernel_initializer=config.get("initializer"))(x)
	x = Activation("relu")(x)

	# Prepare output if last block
	if last_block:
		x = Conv2D(config.get("num_filters_end"), (1, 1),
			kernel_initializer=config.get("initializer"))(x)

	return x


def expansive_path(x, skip_inputs):
	'''
		U-Net expansive path.
		Initializes multiple upsampling blocks for upsampling.
	'''
	num_filters = [compute_number_of_filters(index)\
			for index in range(configuration()\
				.get("num_unet_blocks")-1, 0, -1)]

	skip_max_index = len(skip_inputs) - 1

	for index, block_num_filters in enumerate(num_filters):
		skip_index = skip_max_index - index
		last_block = index == len(num_filters)-1
		x = upconv_block(x, block_num_filters,\
			skip_inputs[skip_index], last_block)

	return x


def build_unet():
	''' Construct U-Net. '''
	config = configuration()
	input_shape = (config.get("input_height"),\
		config.get("input_width"), config.get("input_dim"))

	# Construct input layer
	input_data = Input(shape=input_shape)

	# Construct Contracting path
	contracted_data, skip_inputs = contracting_path(input_data)

	# Construct Expansive path
	expanded_data = expansive_path(contracted_data, skip_inputs)

	# Define model
	model = Model(input_data, expanded_data, name="U-Net")

	return model


def compute_number_of_filters(block_number):
	'''
		Compute the number of filters for a specific
		U-Net block given its position in the contracting path.
	'''
	return configuration().get("num_filters_start") * (2 ** block_number)


'''
	U-NET TRAINING PROCESS BUILDING BLOCKS
'''

def init_model(steps_per_epoch):
	'''
		Initialize a U-Net model.
	'''
	config = configuration()
	model = build_unet()

	# Retrieve compilation input
	loss_init = config.get("loss")(from_logits=True)
	metrics = config.get("metrics")
	num_epochs = config.get("num_epochs")

	# Construct LR schedule
	boundaries = [int(num_epochs * percentage * steps_per_epoch)\
		for percentage in config.get("lr_schedule_percentages")]
	lr_schedule = config.get("lr_schedule_class")(boundaries, config.get("lr_schedule_values"))

	# Init optimizer
	optimizer_init = config.get("optimizer")(learning_rate = lr_schedule)

	# Compile the model
	model.compile(loss=loss_init, optimizer=optimizer_init, metrics=metrics)

	# Plot the model
	plot_model(model, to_file="unet.png")

	# Print model summary
	model.summary()

	return model


def load_dataset():
	'''	Return dataset with info. '''
	config = configuration()

	# Retrieve percentages
	train = config.get("data_train_prc")
	val = config.get("data_val_prc")
	test = config.get("data_test_prc")

	# Redefine splits over full dataset
	splits = [f'train[:{train}%]+test[:{train}%]',\
		f'train[{train}%:{val}%]+test[{train}%:{val}%]',\
		f'train[{val}%:{test}%]+test[{val}%:{test}%]']

	# Return data
	return tfds.load('oxford_iiit_pet:3.*.*', split=splits, data_dir=configuration()\
		.get("dataset_path"), with_info=True) 


def normalize_sample(input_image, input_mask):
	''' Normalize input image and mask class. '''
	# Cast image to float32 and divide by 255
	input_image = tensorflow.cast(input_image, tensorflow.float32) / 255.0

  # Bring classes into range [0, 2]
	input_mask -= 1

	return input_image, input_mask


def preprocess_sample(data_sample):
	''' Resize and normalize dataset samples. '''
	config = configuration()

	# Resize image
	input_image = tensorflow.image.resize(data_sample['image'],\
  	(config.get("input_width"), config.get("input_height")))

  # Resize mask
	input_mask = tensorflow.image.resize(data_sample['segmentation_mask'],\
  	(config.get("mask_width"), config.get("mask_height")))

  # Normalize input image and mask
	input_image, input_mask = normalize_sample(input_image, input_mask)

	return input_image, input_mask


def data_augmentation(inputs, labels):
	''' Perform data augmentation. '''
	# Use the same seed for deterministic randomness over both inputs and labels.
	seed = 36

  # Feed data through layers
	inputs = tensorflow.image.random_flip_left_right(inputs, seed=seed)
	inputs = tensorflow.image.random_flip_up_down(inputs, seed=seed)
	labels = tensorflow.image.random_flip_left_right(labels, seed=seed)
	labels = tensorflow.image.random_flip_up_down(labels, seed=seed)

	return inputs, labels


def compute_sample_weights(image, mask):
	''' Compute sample weights for the image given class. '''
	# Compute relative weight of class
	class_weights = configuration().get("class_weights")
	class_weights = class_weights/tensorflow.reduce_sum(class_weights)

  # Compute same-shaped Tensor as mask with sample weights per
  # mask element. 
	sample_weights = tensorflow.gather(class_weights,indices=\
  	tensorflow.cast(mask, tensorflow.int32))

	return image, mask, sample_weights


def preprocess_dataset(data, dataset_type, dataset_info):
	''' Fully preprocess dataset given dataset type. '''
	config = configuration()
	batch_size = config.get("batch_size")
	buffer_size = config.get("buffer_size")

	# Preprocess data given dataset type.
	if dataset_type == "train" or dataset_type == "val":
		# 1. Perform preprocessing
		# 2. Cache dataset for improved performance
		# 3. Shuffle dataset
		# 4. Generate batches
		# 5. Repeat
		# 6. Perform data augmentation
		# 7. Add sample weights
		# 8. Prefetch new data before it being necessary.
		return (data
				    .map(preprocess_sample)
				    .cache()
				    .shuffle(buffer_size)
				    .batch(batch_size)
				    .repeat()
				    .map(data_augmentation)
				    .map(compute_sample_weights)
				    .prefetch(buffer_size=tensorflow.data.AUTOTUNE))
	else:
		# 1. Perform preprocessing
		# 2. Generate batches
		return (data
						.map(preprocess_sample)
						.batch(batch_size))


def training_callbacks():
	''' Retrieve initialized callbacks for model.fit '''
	return [
		TensorBoard(
		  log_dir=os.path.join(os.getcwd(), "unet_logs"),
		  histogram_freq=1,
		  write_images=True
		)
	]


def probs_to_mask(probs):
	''' Convert Softmax output into mask. '''
	pred_mask = tensorflow.argmax(probs, axis=2)
	return pred_mask


def generate_plot(img_input, mask_truth, mask_probs):
	''' Generate a plot of input, truthy mask and probability mask. '''
	fig, axs = plt.subplots(1, 4)
	fig.set_size_inches(16, 6)

	# Plot the input image
	axs[0].imshow(img_input)
	axs[0].set_title("Input image")

	# Plot the truthy mask
	axs[1].imshow(mask_truth)
	axs[1].set_title("True mask")

	# Plot the predicted mask
	predicted_mask = probs_to_mask(mask_probs)
	axs[2].imshow(predicted_mask)
	axs[2].set_title("Predicted mask")

	# Plot the overlay
	config = configuration()
	img_input_resized = tensorflow.image.resize(img_input, (config.get("mask_width"), config.get("mask_height")))
	axs[3].imshow(img_input_resized)
	axs[3].imshow(predicted_mask, alpha=0.5)
	axs[3].set_title("Overlay")

	# Show the plot
	plt.show()


def main():
	''' Run full training procedure. '''

	# Load config
	config = configuration()
	batch_size = config.get("batch_size")
	validation_sub_splits = config.get("validation_sub_splits")
	num_epochs = config.get("num_epochs")

	# Load data
	(training_data, validation_data, testing_data), info = load_dataset()

	# Make training data ready for model.fit and model.evaluate
	train_batches = preprocess_dataset(training_data, "train", info)
	val_batches = preprocess_dataset(validation_data, "val", info)
	test_batches = preprocess_dataset(testing_data, "test", info)
	
	# Compute data-dependent variables
	train_num_samples = tensorflow.data.experimental.cardinality(training_data).numpy()
	val_num_samples = tensorflow.data.experimental.cardinality(validation_data).numpy()
	steps_per_epoch = train_num_samples // batch_size
	val_steps_per_epoch = val_num_samples // batch_size // validation_sub_splits

	# Initialize model
	model = init_model(steps_per_epoch)

	# Train the model	
	model.fit(train_batches, epochs=num_epochs, batch_size=batch_size,\
		steps_per_epoch=steps_per_epoch, verbose=1,
		validation_steps=val_steps_per_epoch, callbacks=training_callbacks(),\
		validation_data=val_batches)

	# Test the model
	score = model.evaluate(test_batches, verbose=0)
	print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

	# Take first batch from the test images and plot them
	for images, masks in test_batches.take(1):

		# Generate prediction for each image
		predicted_masks = model.predict(images)

		# Plot each image and masks in batch
		for index, (image, mask) in enumerate(zip(images, masks)):
			generate_plot(image, mask, predicted_masks[index])
			if index > 4:
				break


if __name__ == '__main__':
	main()

        
