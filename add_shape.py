
import tensorflow as tf
from keras.applications import MobileNetV3Large
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy

MODEL_NAME = "MobileNetV3Large_E25_B16_A_0.5_D_0.2_AUGFULL_SHUFFLE"
MODEL_PATH = f".\\nets\\MobileNetV3Large\\{MODEL_NAME}"
old_model = tf.keras.models.load_model(MODEL_PATH)

new_model = MobileNetV3Large(classes = 8, weights = None, minimalistic = False, alpha = 0.5, input_shape = (224, 224, 3), dropout_rate = 0.2)
new_model.set_weights(old_model.get_weights())
new_model.compile(loss = CategoricalCrossentropy(), optimizer = Adam(learning_rate = 0.0001), metrics = ['accuracy'])

for layer in new_model.layers:
		if layer is new_model:
			new_model.layers.remove(layer)
			
new_model.save(MODEL_PATH, save_format = 'tf', overwrite = True)