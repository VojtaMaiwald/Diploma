from imgaug import augmenters as iaa
from keras.applications.mobilenet import preprocess_input
from keras.utils.image_utils import load_img, img_to_array
import numpy as np
import cv2 as cv

def augment_images(images):
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

if __name__ == '__main__':
	for q in range(1000):
		images = [img_to_array(load_img(".\\test_imgs\\face.jpg", target_size = (1580, 1049)), dtype = np.uint8)]
		images = augment_images(images)
		#images = np.array([img.astype(np.float32) for img in images])
		images = np.array([img for img in images])
		for i in images:
			i2 = cv.cvtColor(i, cv.COLOR_BGR2RGB)
			#cv.imshow("pic",i2)
			#cv.waitKey(0)
			cv.imwrite(".\\sth\\" + str(q) + ".png", i2)

