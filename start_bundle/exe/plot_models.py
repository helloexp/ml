

from keras.utils import plot_model

from start_bundle.keras_lenet import LeNet

model = LeNet.build(32, 32, 3, 10)

plot_model(model,to_file="./lenet.png", show_shapes=True)

