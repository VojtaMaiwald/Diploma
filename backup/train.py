import os

import numpy as np
from keras import backend as K
from keras.applications import MobileNetV2
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from sklearn.utils import class_weight

from custom_data_generator import CustomDataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# OPTIONS #
IMAGE_SIZE = 224
CLASSIFY = 0
REGRESS = 1
PATH = "./nets/mobilenet.h5"

def root_mean_squared_error(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true)))


def process_data(t_type, paths, labels):
	labels_out = []
	paths_out = []
	count = 0
	for i, (emotion, valence, arousal) in enumerate(labels):
		if t_type == CLASSIFY:
			if emotion > 7:
				continue
			labels_out.append(emotion)
			paths_out.append(paths[i])
		else:
			if arousal == -2 or valence == -2:
				continue
			labels_out.append([valence, arousal])
			paths_out.append(paths[i])
		count += 1
		print('Processed:', count, end='\r')
	if t_type == CLASSIFY:
		weights = class_weight.compute_class_weight('balanced', np.unique(labels_out), labels_out)
		weights = dict(enumerate(weights))
		labels_out = to_categorical(labels_out, num_classes=8)
	else:
		weights = None
	print('Processed:', count)
	return paths_out, labels_out, weights


def mobilenet_v2_model(t_type, dropout=0.5):
	base_model = MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(1024, activation='relu')(x)
	x = Dropout(dropout)(x)
	if t_type == CLASSIFY:
		predictions = Dense(8, activation='softmax')(x)
		model = Model(inputs=base_model.input, outputs=predictions)
		model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00001), metrics=['accuracy'])
	else:
		predictions = Dense(2, activation='linear')(x)
		model = Model(inputs=base_model.input, outputs=predictions)
		model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.00003), metrics=[root_mean_squared_error])

	return model


def regressor_from_classifier(c_model, dropout=False):
	x = c_model.output
	if dropout:
		x = Dropout(0.5)(x)
	predictions = Dense(2, activation='linear', name='regression_output')(x)
	model = Model(inputs=c_model.input, outputs=predictions)
	model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.00003))

	return model


def train(t_type, model, output_path, epochs, batch_size):
	print('** LOADING DATA **')
	t_paths = np.load('training_paths.npy')
	t_labels = np.load('training_labels.npy')
	t_paths, t_labels, t_weights = process_data(t_type, t_paths, t_labels)
	v_paths = np.load('validation_paths.npy')
	v_labels = np.load('validation_labels.npy')
	v_paths, v_labels, v_weights = process_data(t_type, v_paths, v_labels)

	train_generator = CustomDataGenerator(t_paths, t_labels, batch_size, shuffle=True, augment=True)
	val_generator = CustomDataGenerator(v_paths, v_labels, batch_size)

	t_steps = len(t_labels) // batch_size
	v_steps = len(v_labels) // batch_size

	print('** TRAINING MODEL **')
	if t_type == CLASSIFY:
		history = model.fit_generator(
			train_generator,
			steps_per_epoch=t_steps,
			class_weight=t_weights,
			epochs=epochs,
			validation_data=val_generator,
			validation_steps=v_steps,
			callbacks=[
				ModelCheckpoint(output_path + '{epoch:02d}_{val_loss:.3f}_T.h5', monitor='val_acc',
								save_best_only=False,
								save_weights_only=True),
				CSVLogger('log_classification.csv', append=True, separator=';')],
			workers=8,
			use_multiprocessing=True)
	else:
		history = model.fit_generator(
			train_generator,
			steps_per_epoch=t_steps,
			epochs=epochs,
			validation_data=val_generator,
			validation_steps=v_steps,
			callbacks=[
				ModelCheckpoint(output_path + '{epoch:02d}_{val_loss:.4f}_T.h5', save_best_only=False,
								save_weights_only=True),
				CSVLogger('log_regression.csv', append=True, separator=';')],
			workers=8,
			use_multiprocessing=True)

	print('** EXPORTING MODEL **')
	np.save(output_path + '_HIST', history.history)
	for layer in model.layers:
		if t_type(layer) is Dropout:
			model.layers.remove(layer)
	model.save_weights(output_path + '_weights.h5')
	model.save(output_path + '_full_model.h5')


if __name__ == '__main__':
	model = mobilenet_v2_model(CLASSIFY)
	train(CLASSIFY, model, PATH, 8, 8)



	"""
	Training
To train a categorical model invoke the train function in train.py. Following snippets are just examples, feel free to train with different parameters.

If you would like to train a categorical model:

if __name__ == '__main__':
	model = mobilenet_v2_model(CLASSIFY)
	train(CLASSIFY, model, <OUTPUT_PATH>, 10, 64)
This will create an instance of the categorical model and then train it over 10 epochs with batch size of 64.

If you would like to train a dimensional model:

if __name__ == '__main__':
	model = mobilenet_v2_model(REGRESS)
	train(REGRESS, model, <OUTPUT_PATH>, 15, 16)
This will create an instance of the regression model and then train it over 15 epochs with batch size of 16.

Lastly, if you would like to train a dimensional model by using weights from pretrained classification model:

if __name__ == '__main__':
	model = mobilenet_v2_model(REGRESS)
	model.load_weights(<WEIGHTS_PATH>)
	for layer in model.layers:
		if type(layer) is Dropout:
			model.layers.remove(layer)
	regression_model = regressor_from_classifier(model, dropout=True)
	
	train(REGRESS, regression_model, <OUTPUT_PATH>, 10, 16)
This will create an instance of the categorical model and then load the weights of the pretrained one. First thing we need to do is to remove it's dropout layer. Second, by replacing the categorical model's output layer with a 2 neuron linear output we create a regression model. It will train over 10 epochs with batch size of 16.
	
	
	
	
	
	"""