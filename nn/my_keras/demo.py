from tensorflow.contrib.keras import datasets
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Activation,Dropout
import keras


batch_size = 128
num_classes = 10
epochs = 5

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


print(y_train[0])
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_train[0])


def first_model():
    model = Sequential([
        Dense(512, input_shape=(28 * 28,)),
        Dropout(0.2),
        Activation("relu"),

        Dense(512, input_shape=(28 * 28,)),
        Dropout(0.2),
        Activation("relu"),

        Dense(10),
        Dropout(0.2),
        Activation("softmax")
    ])

    model.summary()

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,validation_data=(x_test,y_test))

    score = model.evaluate(x_test, y_test)

    model.save("./mnist_model.m")
    print(score)

if __name__ == '__main__':
    first_model()
