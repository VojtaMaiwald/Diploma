import tensorflow as tf
import cv2 as cv
import numpy as np
import glob
import os
import re

WEBCAM = False
ONE_IMG_TEST = False
MODEL_PATH = ".\\nets\\MobileNetV3Small\\MobileNetV3Small_E25_B16_A_0.75_AUGFULL_SHUFFLE_MINI"
TEST_IMAGES_PATH = "C:\\Users\\Vojta\\DiplomaProjects\\AffectNet\\val_set\\images\\"
TEST_LABELS_PATH = "C:\\Users\\Vojta\\DiplomaProjects\\AffectNet\\val_set\\all_labels_exp.npy"
DICT = {0: "Neutral", 1: "Happiness", 2: "Sadness", 3: "Surprise", 4: "Fear", 5: "Disgust", 6: "Anger", 7: "Contempt", 8: "None", 9: "Uncertain", 10: "No-Face"}

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
	print(f"{MODEL_PATH}\nImages: {len(images_paths_list)}\nErrors: {errors}\nSuccess rate: {evaluation:.3f} %\nConfusion matrix:\n{tf.math.confusion_matrix(labels, predictions)}")

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
	else:
		init()
		model = tf.keras.models.load_model(MODEL_PATH)
		testValDataset(model)
