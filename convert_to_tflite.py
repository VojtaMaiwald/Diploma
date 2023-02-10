import tensorflow as tf

MODEL_PATH = ".\\nets\\MobileNetV2\\MobileNetV2_E25_B8_AUGFULL"
EXPORT_PATH = ".\\nets_tflite\\"

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_PATH) # path to the SavedModel directory
converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
print(converter.inference_input_type, converter.inference_output_type, converter.target_spec.supported_ops)
#converter.inference_input_type = tf.int8  # or tf.uint8
#converter.inference_output_type = tf.int8  # or tf.uint8
tflite_model = converter.convert()

# Save the model.
with open(EXPORT_PATH + "MobileNetV2_E25_B8_AUGFULL_OPT_DEF.tflite", 'wb') as f:
	f.write(tflite_model)