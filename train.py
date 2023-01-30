
import os
import re
import numpy as np
import glob
import tensorflow as tf
from keras import backend as K
from keras.applications import MobileNetV2
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.losses import CategoricalCrossentropy
from sklearn.utils import class_weight
from sequence_loader import SequenceLoader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

#py -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
#ssh mai0042@158.196.109.98

MODEL_PATH = ".\\nets\\MobileNetV2\\"
TRAIN_IMAGES_PATH = "C:\\Users\\Vojta\\DiplomaProjects\\AffectNet\\train_set\\images\\"
TRAIN_LABELS_PATH = "C:\\Users\\Vojta\\DiplomaProjects\\AffectNet\\train_set\\all_labels_exp.npy"
TEST_IMAGES_PATH = "C:\\Users\\Vojta\\DiplomaProjects\\AffectNet\\val_set\\images\\"
TEST_LABELS_PATH = "C:\\Users\\Vojta\\DiplomaProjects\\AffectNet\\val_set\\all_labels_exp.npy"
BATCH_SIZE = 14
EPOCHS = 25
DONE_EPOCHS = 20
DROPOUT = 0.5
IMAGE_SHAPE = (224, 224, 3)
MODEL_NAME = f"MobileNetV2_B128_E25_D0.5"

def init():
	#print(os.getenv("TF_GPU_ALLOCATOR"))

	gpus = tf.config.list_physical_devices('GPU')
	if gpus:
		try:
			# Currently, memory growth needs to be the same across GPUs
			for gpu in gpus:
				tf.config.experimental.set_memory_growth(gpu, True)
			logical_gpus = tf.config.list_logical_devices('GPU')
			print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
		except RuntimeError as e:
			# Memory growth must be set before GPUs have been initialized
			print(e)

def root_mean_squared_error(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true)))

def load_model(existingModelPath = None):
	if existingModelPath != None:
		model = tf.keras.models.load_model(existingModelPath)
	else:
		model = MobileNetV2(classes = 8, weights = None)
		model.compile(loss = CategoricalCrossentropy(), optimizer = Adam(learning_rate = 0.0001), metrics =['accuracy'])

	return model

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def load_dataset(labels_path, images_path):
	
	labels = np.load(labels_path)
	images_paths_list = glob.glob(images_path + "*.jpg")
	images_paths_list.sort(key = natural_keys)
	#augment = False
	#shuffle = False
	#weights = None

	weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(labels), y = labels)
	weights = dict(enumerate(weights))
	labels = to_categorical(labels, num_classes = 8)
	augment = True
	shuffle = False

	sequence = SequenceLoader(images_paths_list, labels, BATCH_SIZE, IMAGE_SHAPE, shuffle, augment)
	return sequence, len(images_paths_list), weights

if __name__ == "__main__":
	init()

	#model = load_model(".\\nets3\\_full_model.tf")
	model = load_model()
	print(" ***** MODEL LOADED ***** ")
	train_sequence, train_labels_count, train_weights = load_dataset(TRAIN_LABELS_PATH, TRAIN_IMAGES_PATH)
	test_sequence, test_labels_count, test_weights = load_dataset(TEST_LABELS_PATH, TEST_IMAGES_PATH)
	print(" ***** SEQUENCES READY ***** ")

	history = model.fit(
		train_sequence,
		steps_per_epoch = train_labels_count // BATCH_SIZE,
		class_weight = train_weights,
		epochs = EPOCHS,
		validation_data = test_sequence,
		validation_steps = test_labels_count // BATCH_SIZE,
		callbacks = [
			#ModelCheckpoint(MODEL_PATH + "MODEL_NAME" + f'_{DONE_EPOCHS+epoch:02d}_{val_loss:.3f}_T.tf', monitor = 'val_acc',
			ModelCheckpoint(MODEL_PATH + MODEL_NAME + '_{epoch:02d}_{val_loss:.3f}_T.tf', monitor = 'val_acc',
							save_best_only = False,
							save_weights_only = False,
							save_format = 'tf'),
			CSVLogger(MODEL_PATH + MODEL_NAME + '_log_classification.csv', append = True, separator = ';')],
		workers = 12,
		use_multiprocessing = False) # False
	"""
	use_multiprocessing:
	Boolean. Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False. Note that because this implementation relies on multiprocessing, you should not pass non-picklable arguments to the generator as they can't be passed easily to children processes.
	"""
	print(" ***** MODEL FITTED ***** ")

	for layer in model.layers:
		if layer is Dropout:
			model.layers.remove(layer)
	model.save_weights(MODEL_PATH + MODEL_NAME + '_weights', save_format = 'tf', overwrite = True)
	model.save(MODEL_PATH + MODEL_NAME + '_full_model', save_format = 'tf', overwrite = True)
	print(" ***** ENDING ***** ")
	np.save(MODEL_PATH + '_HIST', history.history)
	print(history.history["val_acc"])