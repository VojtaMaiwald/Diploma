import numpy as np
import tensorflow as tf
from keras.applications import MobileNetV2
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model

"""
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model_facial_expression_quant.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
print(input_details)
print(output_details)
print(input_shape)

"""

model = MobileNetV2(input_shape = (224, 224, 3), include_top = False, weights = "imagenet", classes = 8)
model.trainable = False
x = model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation = 'relu')(x)
x = Dropout(0.5)(x)

predictions = Dense(8, activation = 'softmax')(x)
model = Model(inputs = model.input, outputs = predictions)
print(model.summary())