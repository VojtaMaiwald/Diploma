# Reading an image using OpenCV
import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from keras.utils.image_utils import load_img, img_to_array

images_array = [img_to_array(load_img("test_imgs\\angry.jpg", target_size = (224, 224)), dtype = np.uint8) for k in range(100)]
print(type(images_array), type(images_array[0]))

sometimes1= lambda aug: iaa.Sometimes(0.1, aug)
sometimes2= lambda aug: iaa.Sometimes(0.02, aug)
# preparing a sequence of functions for augmentation
seq = iaa.Sequential([
    sometimes1(iaa.Fliplr(0.5)), 
    sometimes1(iaa.Crop(percent=(0, 0.1))),
    sometimes1(iaa.LinearContrast((0.6, 1.2))),
    sometimes1(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)),
    sometimes1(iaa.Multiply((0.8, 1.2), per_channel = 0.2)),
    sometimes1(iaa.Rotate((-30, 30))),
    sometimes1(iaa.Affine(translate_percent = {"x": (-0.1, 0.1), "y": (-0.1, 0.1)})),
	sometimes1(iaa.Affine(rotate = (-10, 10))),
    sometimes1(iaa.Resize((0.7, 1.2))),
    sometimes1(iaa.AddToBrightness((-30, 30))),
    sometimes2(iaa.GaussianBlur((0, 2))),
    sometimes2(iaa.MotionBlur((3, 4))),
    ], random_order = True)  

# passing the input to the Sequential function
images_aug = seq(images=images_array)

#images_aug = np.array([img.astype(np.float32) for img in images_aug])

# Display all the augmented images
for img in images_aug:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('Augmented Image', img)
    cv2.waitKey()