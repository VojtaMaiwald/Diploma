import keras
import numpy as np
from imgaug import augmenters as iaa
from keras.applications.mobilenet import preprocess_input
from keras.utils.image_utils import load_img, img_to_array

class SequenceLoader(keras.utils.Sequence):
	def __init__(self, images_paths, labels, batch_size, image_shape, shuffle = False, augment = False):
		"""Constructor"""
		self.images_paths = images_paths            # array of image paths
		self.labels = labels                        # array of label paths
		self.batch_size = batch_size                # batch size
		self.image_shape = image_shape              # image dimensions
		self.shuffle = shuffle                      # shuffle images
		self.augment = augment                      # augment images
		self.indices = range(len(labels))

	def __len__(self):
		"""Denotes the number of batches per epoch"""
		return len(self.images_paths) // self.batch_size
		
	#def on_epoch_end(self): # is it really called?
	#	"""Updates indices after each epoch"""
	#	self.indices = np.arange(len(self.images_paths))
	#	if self.shuffle:
	#		np.random.shuffle(self.indices)

	def __getitem__(self, index):
		"""Generate one batch of data"""
		
		indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]	# selects indices of data for next batch

		# select data and load images
		labels = np.array([self.labels[i] for i in indices])
		images = [img_to_array(load_img(self.images_paths[k], target_size = (224, 224))) for k in indices]
		# preprocess and augment data
		#if self.augment:
		#	images = self.augment_images(images)
		images = np.array([img for img in images])
		#images = np.array([preprocess_input(img) for img in images])
		return images, labels

	def augment_images(self, images):
		"""Apply data augmentation"""
		sometimes = lambda aug: iaa.Sometimes(0.25, aug)
		seq = iaa.Sequential(
			[
				iaa.Fliplr(0.25),													# horizontally flip 50% of images
				sometimes(iaa.Affine(
					translate_percent = {"x": (-0.1, 0.1), "y": (-0.1, 0.1)},		# translate by +-20 % (per axis)
					rotate = (-10, 10),												# rotate by +-10 degrees
				))
			]
		)
		return seq.augment_images(images)