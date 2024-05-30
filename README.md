# Description

This is a repository of source codes for training, evaluating and testing of neural network models for image classification on the AffectNet[[1]](#1)[[2]](#2) dataset. The neural network architectures are mainly taken from the Keras implementation[[3]](#3), but in some cases from other sources[[4]](#4)[[5]](#5)[[6]](#6)[[7]](#7)[[8]](#8). Trained models are also available from this repository - in TensorFlow 2 standard format, as well as converted and optimised models in TensorFlow Lite format.

In total, 63 classification and 28 regression models were trained for comparison and to decide which were suitable for use on mobile devices. For testing purposes on real devices, an Android application was developed[[9]](#9) and released[[10]](#10).

For each classification model there is an average percentage and a confusion matrix. For each regression model there are arousal, valence and average values of RMSE. All based on a test batch of images from the AffectNet dataset.

In my diploma thesis[[11]](#11) you can find more information about the influence of architectural and training parameters of neural networks on total model latency, classification and RMSE results (Czech language only). There is also a scientific publication available in English[[12]](#12) - Facial Emotion Recognition for Mobile Devices: A Practical Review.

# References
<a id="1">[1]</a>
http://mohammadmahoor.com/affectnet/

<a id="2">[2]</a>
http://mohammadmahoor.com/wp-content/uploads/2017/08/AffectNet_oneColumn-2.pdf

<a id="3">[3]</a>
https://keras.io/api/applications/

<a id="4">[4]</a>
https://github.com/YeFeng1993/GhostNet-Keras

<a id="5">[5]</a>
https://github.com/abhoi/Keras-MnasNet

<a id="6">[6]</a>
https://github.com/Haikoitoh/paper-implementation/blob/main/ShuffleNet.ipynb

<a id="7">[7]</a>
https://github.com/opconty/keras-shufflenetV2

<a id="8">[8]</a>
https://github.com/cmasch/squeezenet/blob/master/squeezenet.py

<a id="9">[9]</a>
https://github.com/VojtaMaiwald/FaceEmotionRecognitionTest

<a id="10">[10]</a>
https://play.google.com/store/apps/details?id=cz.vsb.faceemotionrecognition

<a id="11">[11]</a>
https://dspace.vsb.cz/handle/10084/151688

<a id="12">[12]</a>
https://ieeexplore.ieee.org/document/10414102
