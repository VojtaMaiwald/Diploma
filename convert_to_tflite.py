import tensorflow as tf
import glob

for name in glob.glob(".\\nets\\EfficientNetB0\\EfficientNet*") + glob.glob(".\\nets\\MobileNetV*\\MobileNetV*") + glob.glob(".\\nets\\NASNetMobile\\NASNetMobile*"):

	#IMPORT_MODEL_PATH = ".\\nets\\EfficientNetB0\\"
	#IMPORT_MODEL_NAME = "EfficientNetB0_E25_B4_AUGFULL_SHUFFLE"
	IMPORT_MODEL_NAME = name.split("\\")[-1]

	EXPORT_MODEL_PATH = ".\\nets_tflite\\"
	EXPORT_MODEL_NAME = f"{IMPORT_MODEL_NAME}_float16.tflite"

	# Convert the model
	converter = tf.lite.TFLiteConverter.from_saved_model(name)
	#converter = tf.lite.TFLiteConverter.from_saved_model(IMPORT_MODEL_PATH + IMPORT_MODEL_NAME)
	#converter = tf.lite.TFLiteConverter.from_keras_model(IMPORT_MODEL_PATH + IMPORT_MODEL_NAME)

	#print(converter.inference_input_type, converter.inference_output_type, converter.target_spec.supported_ops, converter.target_spec.supported_types)

	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	converter.target_spec.supported_types = [tf.float16, tf.float32]

	#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
	converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]

	#converter.inference_input_type = tf.int8  # or tf.uint8
	#converter.inference_output_type = tf.int8  # or tf.uint8
	tflite_model = converter.convert()

	# Save the model.
	with open(EXPORT_MODEL_PATH + EXPORT_MODEL_NAME, 'wb') as f:
		f.write(tflite_model)

	#interpreter = tf.lite.Interpreter(model_path = EXPORT_MODEL_PATH + EXPORT_MODEL_NAME)
	#input_shape = interpreter.get_input_details()[0]["shape"]
	#input_type = interpreter.get_input_details()[0]["dtype"]
	#output_shape = interpreter.get_output_details()[0]["shape"]
	#output_type = interpreter.get_output_details()[0]["dtype"]
	#
	#print(f"Input shape:\t{input_shape}\nInput type:\t{input_type}\nOutput shape:\t{output_shape}\nOutput type:\t{output_type}")