from keras.utils import plot_model
from keras.applications import MobileNetV2

model = MobileNetV2(classes = 8, weights = None)

plot_model(model, to_file="model.png", show_shapes=True)