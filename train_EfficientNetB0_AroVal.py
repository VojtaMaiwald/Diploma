
import os
import re
import numpy as np
import glob
import tensorflow as tf
from keras import backend as K
from keras.applications import EfficientNetB0
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.utils.np_utils import to_categorical
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from sklearn.utils import class_weight
from sequence_loader import SequenceLoader
from keras import backend as K
import cv2 as cv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

#py -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

#MODEL_PATH = "./nets/EfficientNet/"
#TRAIN_IMAGES_PATH = "/sp1/train_set/images/"
#TRAIN_LABELS_PATH = "/sp1/train_set/all_labels_exp.npy"
#TEST_IMAGES_PATH = "/sp1/val_set/images/"
#TEST_LABELS_PATH = "/sp1/val_set/all_labels_exp.npy"

MODEL_PATH = ".\\nets\\EfficientNet\\"
TRAIN_IMAGES_PATH = "C:\\Users\\Vojta\\DiplomaProjects\\AffectNet\\train_set\\images\\"
TRAIN_ARO_LABELS_PATH = "C:\\Users\\Vojta\\DiplomaProjects\\AffectNet\\train_set\\all_labels_aro.npy"
TRAIN_VAL_LABELS_PATH = "C:\\Users\\Vojta\\DiplomaProjects\\AffectNet\\train_set\\all_labels_val.npy"
TEST_IMAGES_PATH = "C:\\Users\\Vojta\\DiplomaProjects\\AffectNet\\val_set\\images\\"
TEST_ARO_LABELS_PATH = "C:\\Users\\Vojta\\DiplomaProjects\\AffectNet\\val_set\\all_labels_aro.npy"
TEST_VAL_LABELS_PATH = "C:\\Users\\Vojta\\DiplomaProjects\\AffectNet\\val_set\\all_labels_val.npy"


BATCH_SIZE = 8 # BATCH_SIZE * strategy.num_replicas_in_sync
EPOCHS = 10
IMAGE_SHAPE = (224, 224, 3)
AUGMENT = True
SHUFFLE = True
LEARNING_RATE = 0.0001
DROPOUT = 0.2
ENDING_STRING = ("AUGFULL" if AUGMENT else "") + ("_SHUFFLE" if SHUFFLE else "")
MODEL_NAME = f"EfficientNetB0_AroVal_E{EPOCHS}_B{BATCH_SIZE}_SGD{LEARNING_RATE}_{ENDING_STRING}"

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
	#strategy = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.ReductionToOneDevice())
	#return strategy

#def load_model(strategy, existingModelPath = None):
def load_model(existingModelPath = None):
	if existingModelPath != None:
		model = tf.keras.models.load_model(existingModelPath)
	else:
		 #with strategy.scope():
			base_model = EfficientNetB0(include_top = False, weights = None, input_shape = IMAGE_SHAPE)
			x = base_model.output
			x = GlobalAveragePooling2D()(x)
			x = Dense(1024, activation = 'relu')(x)
			x = Dropout(DROPOUT)(x)
			predictions = Dense(2, activation = 'linear')(x)
			model = Model(inputs = base_model.input, outputs = predictions)
			model.compile(loss = MeanSquaredError(), optimizer = Adam(learning_rate = LEARNING_RATE), metrics = [RootMeanSquaredError()])

	return model

def atoi(text):
	return int(text) if text.isdigit() else text

def natural_keys(text):
	return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def load_dataset(aro_labels_path, val_labels_path, images_path, train = True):
	labels_aro = np.load(aro_labels_path)
	labels_val = np.load(val_labels_path)
	images_paths_list = glob.glob(images_path + "*.jpg")
	images_paths_list.sort(key = natural_keys)
	labels = [[labels_aro[i], labels_val[i]] for i in range(len(images_paths_list))]

	sequence = SequenceLoader(images_paths_list, labels, BATCH_SIZE, IMAGE_SHAPE, SHUFFLE and train, AUGMENT and train)
	return sequence, len(images_paths_list)

if __name__ == "__main__":
	#strategy = init()
	init()
	#model = load_model(strategy)
	model = load_model()
	print(" ***** MODEL LOADED ***** ")
	train_sequence, train_labels_count = load_dataset(TRAIN_ARO_LABELS_PATH, TRAIN_VAL_LABELS_PATH, TRAIN_IMAGES_PATH)
	test_sequence, test_labels_count = load_dataset(TEST_ARO_LABELS_PATH, TEST_VAL_LABELS_PATH, TEST_IMAGES_PATH, False)
	print(" ***** SEQUENCES READY ***** ")

	history = model.fit(
		train_sequence,
		steps_per_epoch = train_labels_count // BATCH_SIZE,
		epochs = EPOCHS,
		validation_data = test_sequence,
		validation_steps = test_labels_count // BATCH_SIZE,
		callbacks = [
			ModelCheckpoint(MODEL_PATH + MODEL_NAME + '_E_{epoch:02d}_{val_loss:.3f}_T.tf',
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
	
	f = open(MODEL_PATH + MODEL_NAME + "/stats.txt", "w")
	f.write("root_mean_squared_error:\n")
	f.write(str(history.history['root_mean_squared_error']))
	f.write("\n")
	f.write("val_root_mean_squared_error:\n")
	f.write(str(history.history['val_root_mean_squared_error']))
	f.write("\n\n")

	model = tf.keras.models.load_model(MODEL_PATH + MODEL_NAME)
	labels_aro = np.load(TEST_ARO_LABELS_PATH)
	labels_val = np.load(TEST_VAL_LABELS_PATH)
	images_paths_list = glob.glob(TEST_IMAGES_PATH + "*.jpg")
	images_paths_list.sort(key = natural_keys)
	labels = [[labels_aro[i], labels_val[i]] for i in range(len(images_paths_list))]
	RMSE_avg_aro = 0
	RMSE_avg_val = 0
	file_string = ""

	for i in range(len(images_paths_list)):
		img_path = images_paths_list[i]
		img = cv.imread(img_path, 1)
		img = img.reshape(1, 224, 224, 3)

		aro_pred, val_pred = model.predict(img, verbose = 0)[0]
		aro_label, val_label = labels[i]
		RMSE_avg_aro += (aro_pred - aro_label) ** 2
		RMSE_avg_val += (val_pred - val_label) ** 2

		file_string += f"\n{aro_label:.8f}\t{aro_pred:.8f}\t{val_label:.8f}\t{val_pred:.8f}"

		print(f"{i} / {len(images_paths_list)}\t\tArousal avg RMSE: {(np.sqrt((1 / (i + 1)) * RMSE_avg_aro)):.4f}\t\tValence avg RMSE: {(np.sqrt((1 / (i + 1)) * RMSE_avg_val)):.4f}        ", end = "\r")

	RMSE_avg_aro = np.sqrt((1 / len(images_paths_list)) * RMSE_avg_aro)
	RMSE_avg_val = np.sqrt((1 / len(images_paths_list)) * RMSE_avg_val)

	print("\n")
	print(f"{MODEL_NAME}\nImages: {len(images_paths_list)}\nArousal average RMSE: {RMSE_avg_aro:.4f}\nValence average RMSE: {RMSE_avg_val:.4f}\nAverage total RMSE: {((RMSE_avg_aro + RMSE_avg_val) / 2):.4f}")
	f.write(f"{MODEL_NAME}\nImages: {len(images_paths_list)}\nArousal average RMSE: {RMSE_avg_aro:.8f}\nValence average RMSE: {RMSE_avg_val:.8f}\nAverage total RMSE: {((RMSE_avg_aro + RMSE_avg_val) / 2):.8f}")
	f.write("\n\n")
	f.write("GT_aro\tpred_aro\tGT_val\tpred_val")
	f.write(file_string)
	f.close()
	print(" ***** STATS SAVED ***** ")