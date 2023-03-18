import tensorflow as tf

IMPORT_MODEL_PATH = ".\\nets\\MobileNetV2\\"
IMPORT_MODEL_NAME = "MobileNetV2_E25_B8_AUGFULL"

EXPORT_MODEL_PATH = ".\\nets_tflite\\"
EXPORT_MODEL_NAME = f"{IMPORT_MODEL_NAME}.tflite"

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(IMPORT_MODEL_PATH)
#converter = tf.lite.TFLiteConverter.from_keras_model(IMPORT_MODEL_PATH)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
print(converter.inference_input_type, converter.inference_output_type, converter.target_spec.supported_ops)
#converter.inference_input_type = tf.int8  # or tf.uint8
#converter.inference_output_type = tf.int8  # or tf.uint8
tflite_model = converter.convert()

# Save the model.
with open(EXPORT_MODEL_PATH + "MobileNetV2_E25_B8_AUGFULL_OPT_DEF.tflite", 'wb') as f:
	f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_path = EXPORT_MODEL_PATH + EXPORT_MODEL_NAME)
input_shape = interpreter.get_input_details()[0]["shape"]
input_type = interpreter.get_input_details()[0]["dtype"]
output_shape = interpreter.get_output_details()[0]["shape"]
output_type = interpreter.get_output_details()[0]["dtype"]

print(f"Input shape:\t{input_shape}\nInput type:\t{input_type}\nOutput shape:\t{output_shape}\nOutput type:\t{output_type}")