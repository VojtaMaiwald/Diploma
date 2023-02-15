
import os
import re
import numpy as np
import glob
import tensorflow as tf
from keras import backend as K
from keras.applications import MobileNetV3Large
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.losses import CategoricalCrossentropy
from sklearn.utils import class_weight
from sequence_loader import SequenceLoader
from keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

#py -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

MODEL_PATH = "./nets/MobileNetV3Small/"
TRAIN_IMAGES_PATH = "/sp1/train_set/images/"
TRAIN_LABELS_PATH = "/sp1/train_set/all_labels_exp.npy"
TEST_IMAGES_PATH = "/sp1/val_set/images/"
TEST_LABELS_PATH = "/sp1/val_set/all_labels_exp.npy"
BATCH_SIZE = 16 * 3 # BATCH_SIZE * strategy.num_replicas_in_sync
EPOCHS = 25
DONE_EPOCHS = 20
DROPOUT = 0.5
IMAGE_SHAPE = (224, 224, 3)
ALPHA = 0.5
MINIMALISTIC = True
MODEL_NAME = "MobileNetV3Large_E25_B16_A_0.5_D_0.5_AUGFULL_SHUFFLE_MINI"

def init():
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

	#strategy = tf.distribute.MultiWorkerMirroredStrategy()
	#strategy = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())
	#strategy = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.NcclAllReduce())
	strategy = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.ReductionToOneDevice())
	return strategy

def root_mean_squared_error(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true)))

def load_model(strategy, existingModelPath = None):
#def load_model(existingModelPath = None):
	if existingModelPath != None:
		model = tf.keras.models.load_model(existingModelPath)
	else:
		with strategy.scope():
			model = MobileNetV3Large(classes = 8, weights = None, minimalistic = MINIMALISTIC, alpha = ALPHA, dropout_rate = DROPOUT)
			model.compile(loss = CategoricalCrossentropy(), optimizer = Adam(learning_rate = 0.0001), metrics = ['accuracy'])

	return model

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def load_dataset(labels_path, images_path):
	
	labels = np.load(labels_path)
	images_paths_list = glob.glob(images_path + "*.jpg")
	images_paths_list.sort(key = natural_keys)

	weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(labels), y = labels)
	weights = dict(enumerate(weights))
	labels = to_categorical(labels, num_classes = 8)
	augment = True
	shuffle = True

	sequence = SequenceLoader(images_paths_list, labels, BATCH_SIZE, IMAGE_SHAPE, shuffle, augment)
	return sequence, len(images_paths_list), weights

if __name__ == "__main__":
	strategy = init()
	#init()
	model = load_model(strategy)
	#model = load_model()
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
			ModelCheckpoint(MODEL_PATH + MODEL_NAME + '_E_{epoch:02d}_{val_loss:.3f}_T.tf', monitor = 'val_acc',
							save_best_only = False,
							save_weights_only = False,
							save_format = 'tf'),
			],
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
	model.save(MODEL_PATH + MODEL_NAME, save_format = 'tf', overwrite = True)
	print(" ***** ENDING ***** ")
	np.save(MODEL_PATH + '_HIST', history.history)
	#print("accuracy:\n", history.history['accuracy'])
	#print("val_accuracy:\n", history.history['val_accuracy'])
	f = open(MODEL_PATH + MODEL_NAME + "/stats.txt", "w")
	f.write("accuracy:\n")
	f.write(str(history.history['accuracy']))
	f.write("\n")
	f.write("val_accuracy:\n")
	f.write(str(history.history['val_accuracy']))
	f.write("\n\n")
	f.close()
	print(" ***** STATS SAVED ***** ")