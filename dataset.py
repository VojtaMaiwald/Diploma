
import numpy as np
import glob
from keras.utils.np_utils import to_categorical
from sklearn.utils import class_weight
from sequence_loader import SequenceLoader
import cv2 as cv
import re

TRAIN_IMAGES_PATH = "C:\\Users\\Vojta\\DiplomaProjects\\AffectNet\\train_set\\images\\"
TRAIN_LABELS_PATH = "C:\\Users\\Vojta\\DiplomaProjects\\AffectNet\\train_set\\all_labels_exp.npy"
TEST_IMAGES_PATH = "C:\\Users\\Vojta\\DiplomaProjects\\AffectNet\\val_set\\images\\"
TEST_LABELS_PATH = "C:\\Users\\Vojta\\DiplomaProjects\\AffectNet\\val_set\\all_labels_exp.npy"

DICT = {0: "Neutral", 1: "Happy", 2: "Sad", 3: "Surprise", 4: "Fear", 5: "Disgust", 6: "Anger", 7: "Contempt"}

BATCH_SIZE = 14
EPOCHS = 22
DROPOUT = 0.5
IMAGE_SHAPE = (224, 224, 3)


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
		

	def __getitem__(self, index):
		"""Generate one batch of data"""
		
		indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]	# selects indices of data for next batch

		# select data and load images
		labels = np.array([self.labels[i] for i in indices])
		
		return [self.images_paths[k] for k in indices], labels


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def load_dataset(labels_path, images_path):
	
	labels = np.load(labels_path)
	print(labels[0:14])
	images_paths_list = sorted(glob.glob(images_path + "*.jpg"), key =  natural_keys)

	weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(labels), y = labels)
	weights = dict(enumerate(weights))
	labels = to_categorical(labels, num_classes = 8)
	augment = True
	shuffle = False

	sequence = SequenceLoader(images_paths_list, labels, BATCH_SIZE, IMAGE_SHAPE, shuffle, augment)
	return sequence, len(images_paths_list), weights


if __name__ == "__main__":
	train_sequence, train_labels_count, train_weights = load_dataset(TRAIN_LABELS_PATH, TRAIN_IMAGES_PATH)

	quit = False
	for j in range(train_labels_count):
		imgs, labels = train_sequence.__getitem__(j)
		print(labels)
		print("-------------------")
		for i in zip(imgs, labels):
			img = cv.imread(i[0])
			emotion = DICT[np.argmax(i[1])]
			cv.putText(img, emotion, (10, 210), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)
			cv.putText(img, emotion, (10, 210), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
			cv.imshow("img", img)
			
			if cv.waitKey(0) & 0xFF == ord('q'):
				quit = True
				break
		if quit:
			break