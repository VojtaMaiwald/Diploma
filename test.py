import tensorflow as tf
import cv2 as cv
import numpy as np
import glob
import os
import re

MODEL_NAME = "ShuffleNetV2_AroVal_E10_B8_SC1.5_BOTTLENECK1_SGD0.01_AUGFULL_SHUFFLE"
MODEL_PATH = f".\\nets\\ShuffleNetV2\\{MODEL_NAME}"
TEST_IMAGES_PATH = "C:\\Users\\Vojta\\DiplomaProjects\\AffectNet\\val_set\\images\\"
TEST_LABELS_PATH = "C:\\Users\\Vojta\\DiplomaProjects\\AffectNet\\val_set\\all_labels_exp.npy"
DICT = {0: "Neutral", 1: "Happiness", 2: "Sadness", 3: "Surprise", 4: "Fear", 5: "Disgust", 6: "Anger", 7: "Contempt", 8: "None", 9: "Uncertain", 10: "No-Face"}
TEST_ARO_LABELS_PATH = "C:\\Users\\Vojta\\DiplomaProjects\\AffectNet\\val_set\\all_labels_aro.npy"
TEST_VAL_LABELS_PATH = "C:\\Users\\Vojta\\DiplomaProjects\\AffectNet\\val_set\\all_labels_val.npy"

WEBCAM = False
ONE_IMG_TEST = False
REGRESSION = True

def webcam(model):
	cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
	capture = cv.VideoCapture(0)

	while (True):
		ret, image = capture.read()
		rectangles = cascade.detectMultiScale(image, 1.3, 3)

		for rect in rectangles:
			img = image[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]
			img = cv.resize(img, (224, 224))
			img = img.reshape(1, 224, 224, 3)
			prediction = model.predict(img)[0]
			print(prediction)
			cv.rectangle(image, rect, (0, 0, 0), 3)
			cv.rectangle(image, rect, (255, 255, 255), 1)
			cv.putText(image, DICT[np.argmax(prediction)], (rect[1], rect[0]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, 2)
			cv.putText(image, DICT[np.argmax(prediction)], (rect[1], rect[0]), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, 2)

		cv.imshow("webcam", image)
		if cv.waitKey(1) & 0xFF == ord('q'):
			break

def testOneImage(model):
	img = cv.imread("test_imgs\\angry.jpg", 1)
	cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
	rect = cascade.detectMultiScale(img, 1.3, 3)[0]
	img = img[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]
	img = cv.resize(img, (224, 224))
	img = img.reshape(1, 224, 224, 3)
	prediction = model.predict(img)[0]
	print(DICT[np.argmax(prediction)], prediction[np.argmax(prediction)], prediction)

def testValDataset(model):
	labels = np.load(TEST_LABELS_PATH)
	predictions = []
	images_paths_list = glob.glob(TEST_IMAGES_PATH + "*.jpg")
	images_paths_list.sort(key = natural_keys)
	errors = 0

	for i in range(len(images_paths_list)):
		img_path = images_paths_list[i]
		img = cv.imread(img_path, 1)
		img = img.reshape(1, 224, 224, 3)
		prediction = model.predict(img, verbose = 0)[0]
		predictions.append(np.argmax(prediction))
		if np.argmax(prediction) != labels[i]:
			errors += 1
		evaluation = (1 - (errors / (i + 1))) * 100
		print(f"{i} / {len(images_paths_list)}\t\tSuccess rate: {evaluation:.3f} %        ", end = "\r")

	print("\n")
	evaluation = (1 - (errors / (len(images_paths_list)))) * 100
	print(f"{MODEL_NAME}\nImages: {len(images_paths_list)}\nErrors: {errors}\nSuccess rate: {evaluation:.3f} %\nConfusion matrix:\n{tf.math.confusion_matrix(labels, predictions)}")

def testValDatasetRegression(model):
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

	f = open(MODEL_PATH + "\\stats2.txt", "w")
	print(f"{MODEL_NAME}\nImages: {len(images_paths_list)}\nArousal average RMSE: {RMSE_avg_aro:.4f}\nValence average RMSE: {RMSE_avg_val:.4f}\nAverage total RMSE: {((RMSE_avg_aro + RMSE_avg_val) / 2):.4f}")
	f.write(f"{MODEL_NAME}\nImages: {len(images_paths_list)}\nArousal average RMSE: {RMSE_avg_aro:.8f}\nValence average RMSE: {RMSE_avg_val:.8f}\nAverage total RMSE: {((RMSE_avg_aro + RMSE_avg_val) / 2):.8f}")
	f.write("\n\n")
	f.write("GT_aro\tpred_aro\tGT_val\tpred_val")
	f.write(file_string)
	f.close()

def atoi(text):
	return int(text) if text.isdigit() else text

def natural_keys(text):
	return [ atoi(c) for c in re.split(r'(\d+)', text) ]

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

if __name__ == '__main__':
	if WEBCAM:
		model = tf.keras.models.load_model(MODEL_PATH)
		webcam(model)
	elif ONE_IMG_TEST:
		init()
		model = tf.keras.models.load_model(MODEL_PATH)
		testOneImage(model)
	elif REGRESSION:
		init()
		model = tf.keras.models.load_model(MODEL_PATH)
		testValDatasetRegression(model)
	else:
		init()
		model = tf.keras.models.load_model(MODEL_PATH)
		testValDataset(model)
