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
		
	def on_epoch_end(self): # is it really called?
		"""Updates indices after each epoch"""
		self.indices = np.arange(len(self.images_paths))
		if self.shuffle:
			np.random.shuffle(self.indices)

	def __getitem__(self, index):
		"""Generate one batch of data"""
		indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]	# selects indices of data for next batch
		# select data and load images
		labels = np.array([self.labels[i] for i in indices])
		#images = [img_to_array(load_img(self.images_paths[k], target_size = (224, 224)), dtype = np.uint8) for k in indices]
		images = [img_to_array(load_img(self.images_paths[k], target_size = (224, 224))) for k in indices]
		# preprocess and augment data
		if self.augment:
			images = self.augment_images(images)
		#images = np.array([img.astype(np.float32) for img in images])
		images = np.array([img for img in images])
		#images = images.astype(np.float32)
		#images = np.array([preprocess_input(img) for img in images])
		return images, labels

	def augment_images(self, images):
		"""Apply data augmentation"""
		sometimes1 = lambda aug: iaa.Sometimes(0.1, aug)
		sometimes2 = lambda aug: iaa.Sometimes(0.02, aug)
		
		# commented augmenters are not compatible with dtype np.float32
		seq = iaa.Sequential([
			sometimes1(iaa.Fliplr(0.5)), 
			sometimes1(iaa.Crop(percent = (0, 0.1))),
			sometimes1(iaa.LinearContrast((0.6, 1.2))),
			sometimes1(iaa.AdditiveGaussianNoise(loc = 0, scale = (0.0, 0.05 * 255), per_channel = 0.5)),
			sometimes1(iaa.Multiply((0.8, 1.2), per_channel = 0.2)),
			sometimes1(iaa.Rotate((-30, 30))),
			sometimes1(iaa.Affine(translate_percent = {"x": (-0.1, 0.1), "y": (-0.1, 0.1)})),
			sometimes1(iaa.Affine(rotate = (-10, 10))),
			sometimes2(iaa.GaussianBlur((0, 2))),
			sometimes2(iaa.MotionBlur((3, 4))),
			], random_order = True)

		return seq.augment_images(images)
		#return seq(images = images)