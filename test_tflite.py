import numpy as np
import tensorflow as tf
import cv2 as cv
import glob
import re

MODELS = [
	#'EfficientNetB0_E25_B16_AUGFULL_SHUFFLE_float16.tflite',
	#'EfficientNetB0_E25_B4_AUGFULL_SHUFFLE_float16.tflite',
	#'EfficientNetB0_E25_B4_AUGFULL_SHUFFLE_lr_00001_float16.tflite',
	#'EfficientNetB0_E25_B8_AUGFULL_SHUFFLE_float16.tflite',
	#"EfficientNetB0_E25_B4_SGD_0.01_AUGFULL_SHUFFLE_float16.tflite"
#
	#'MobileNetV2_E25_B16_D_0.2_AUGFULL_SHUFFLE_float16.tflite',
	#'MobileNetV2_E25_B32_D_0.2_AUGFULL_float16.tflite',
	#'MobileNetV2_E25_B32_D_0.2_AUGFULL_SHUFFLE_float16.tflite',
	#'MobileNetV2_E25_B32_D_0.2_AUG_float16.tflite',
	#'MobileNetV2_E25_B32_D_0.2_NOAUG_float16.tflite',
	#'MobileNetV2_E25_B8_D_0.2_AUGFULL_float16.tflite',
#
	'MobileNetV3Large_E25_B16_A_0.5_D_0.2_AUGFULL_SHUFFLE_float16.tflite',
	'MobileNetV3Large_E25_B16_A_0.5_D_0.2_AUGFULL_SHUFFLE_MINI_float16.tflite',
	'MobileNetV3Large_E25_B16_A_0.5_D_0.5_AUGFULL_SHUFFLE_float16.tflite',
	'MobileNetV3Large_E25_B16_A_0.5_D_0.5_AUGFULL_SHUFFLE_MINI_float16.tflite',
	'MobileNetV3Large_E25_B16_A_0.75_D_0.2_AUGFULL_SHUFFLE_float16.tflite',
	'MobileNetV3Large_E25_B16_A_0.75_D_0.2_AUGFULL_SHUFFLE_MINI_float16.tflite',
	'MobileNetV3Large_E25_B16_A_1.25_D_0.2_AUGFULL_SHUFFLE_float16.tflite',
	'MobileNetV3Large_E25_B16_A_1.25_D_0.2_AUGFULL_SHUFFLE_MINI_float16.tflite',
	'MobileNetV3Large_E25_B16_A_1.25_D_0.5_AUGFULL_SHUFFLE_float16.tflite',
	'MobileNetV3Large_E25_B16_A_1.25_D_0.5_AUGFULL_SHUFFLE_MINI_float16.tflite',
	'MobileNetV3Large_E25_B16_A_1_D_0.2_AUGFULL_SHUFFLE_float16.tflite',
	'MobileNetV3Large_E25_B16_A_1_D_0.2_AUGFULL_SHUFFLE_MINI_float16.tflite',
#
	#'MobileNetV3Small_E25_B16_A_0.75_D_0.2_AUGFULL_SHUFFLE_float16.tflite',
	#'MobileNetV3Small_E25_B16_A_0.75_D_0.2_AUGFULL_SHUFFLE_MINI_float16.tflite',
	#'MobileNetV3Small_E25_B16_A_1.25_D_0.2_AUGFULL_SHUFFLE_float16.tflite',
	#'MobileNetV3Small_E25_B16_A_1.25_D_0.2_AUGFULL_SHUFFLE_MINI_float16.tflite',
	#'MobileNetV3Small_E25_B32_A_1.5_D_0.2_AUGFULL_SHUFFLE_float16.tflite',
	#'MobileNetV3Small_E30_B16_A_1.25_D_0.5_AUGFULL_SHUFFLE_float16.tflite',
]

MODEL_NAME = "EfficientNetB0_E25_B4_AUGFULL_SHUFFLE_float16.tflite"
MODEL_PATH = f".\\nets_tflite\\{MODEL_NAME}"
DICT = {0: "Neutral", 1: "Happiness", 2: "Sadness", 3: "Surprise", 4: "Fear", 5: "Disgust", 6: "Anger", 7: "Contempt", 8: "None", 9: "Uncertain", 10: "No-Face"}
TEST_IMAGES_PATH = "C:\\Users\\Vojta\\DiplomaProjects\\AffectNet\\val_set\\images\\"
TEST_LABELS_PATH = "C:\\Users\\Vojta\\DiplomaProjects\\AffectNet\\val_set\\all_labels_exp.npy"

TEST_ONE_IMG = False
WEBCAM = False

def testOneImage():
	# Load TFLite model and allocate tensors.
	interpreter = tf.lite.Interpreter(model_path = MODEL_PATH)
	interpreter.allocate_tensors()

	# Get input and output tensors.
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	# Test model on image
	#img = cv.imread("test_imgs\\angry.jpg", 1)
	#img = cv.imread("test_imgs\\happy.jpg", 1)
	#img = cv.imread("test_imgs\\happy2.jpg", 1)
	#img = cv.imread("test_imgs\\sad.jpg", 1)
	img = cv.imread("test_imgs\\surprise.jpg", 1)
	cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
	rect = cascade.detectMultiScale(img, 1.3, 3)[0]
	img = img[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]
	img = cv.resize(img, (224, 224))
	img = img.reshape(1, 224, 224, 3)
	img = np.array(img, dtype = np.float32)
	#img = img / 255.0

	interpreter.set_tensor(input_details[0]['index'], img)
	interpreter.invoke()
	output = interpreter.get_tensor(output_details[0]["index"])[0]
	#print(output)
	print(DICT[np.argmax(output)], output[np.argmax(output)], output)

def webcamTest():
	cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
	capture = cv.VideoCapture(0)

	while (True):
		ret, image = capture.read()
		rectangles = cascade.detectMultiScale(image, 1.3, 3)
		
		# Load TFLite model and allocate tensors.
		interpreter = tf.lite.Interpreter(model_path = MODEL_PATH)
		interpreter.allocate_tensors()

		# Get input and output tensors.
		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()

		for rect in rectangles:
			img = image[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]
			img = cv.resize(img, (224, 224))
			img = img.reshape(1, 224, 224, 3)
			img = np.array(img, dtype = np.float32)
			interpreter.set_tensor(input_details[0]['index'], img)
			interpreter.invoke()
			output = interpreter.get_tensor(output_details[0]["index"])[0]
			emotion = DICT[np.argmax(output)]
			prediction = output[np.argmax(output)]
			print(f"{(prediction * 100):.3f} %\t{emotion}          ", end = "\r")
			cv.rectangle(image, rect, (0, 0, 0), 3)
			cv.rectangle(image, rect, (255, 255, 255), 1)
			cv.putText(image, emotion, (rect[1], rect[0]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, 2)
			cv.putText(image, emotion, (rect[1], rect[0]), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, 2)

		cv.imshow("webcam", image)
		if cv.waitKey(1) & 0xFF == ord('q'):
			break

def testValDataset():
	labels = np.load(TEST_LABELS_PATH)
	predictions = []
	images_paths_list = glob.glob(TEST_IMAGES_PATH + "*.jpg")
	images_paths_list.sort(key = natural_keys)
	errors = 0

	# Load TFLite model and allocate tensors.
	interpreter = tf.lite.Interpreter(model_path = MODEL_PATH, num_threads = 8)
	interpreter.allocate_tensors()

	# Get input and output tensors.
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	for i in range(len(images_paths_list)):
		img_path = images_paths_list[i]
		img = cv.imread(img_path, 1)
		
		img = img.reshape(1, 224, 224, 3)
		img = np.array(img, dtype = np.float32)

		interpreter.set_tensor(input_details[0]['index'], img)
		interpreter.invoke()
		prediction = interpreter.get_tensor(output_details[0]["index"])[0]
		predictions.append(np.argmax(prediction))
		if np.argmax(prediction) != labels[i]:
			errors += 1
		evaluation = (1 - (errors / (i + 1))) * 100
		print(f"{i} / {len(images_paths_list)}\t\tSuccess rate: {evaluation:.3f} %        ", end = "\r")

	evaluation = (1 - (errors / (len(images_paths_list)))) * 100
	print(f"\nImages: {len(images_paths_list)}\nErrors: {errors}\nSuccess rate: {evaluation:.3f} %")
	print(f"\nConfusion matrix:\n {tf.math.confusion_matrix(labels, predictions)}")

	f = open(".\\nets_tflite\\stats_classifier.txt", "a")
	f.write(f"{MODEL_NAME}\nSuccess rate: {evaluation:.3f} %\nConfusion matrix:\n{tf.math.confusion_matrix(labels, predictions)}\n\n")
	f.close()

def atoi(text):
	return int(text) if text.isdigit() else text

def natural_keys(text):
	return [ atoi(c) for c in re.split(r'(\d+)', text) ]

if __name__ == '__main__':

	if TEST_ONE_IMG:
		testOneImage()
	elif WEBCAM:
		webcamTest()
	else:
		for i in MODELS:
			MODEL_NAME = i
			MODEL_PATH = f".\\nets_tflite\\{MODEL_NAME}"
			testValDataset()
