import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Sequential
from keras.layers import Input,Dense,Dropout,Flatten,Activation
from keras.layers import Convolution2D,MaxPooling2D
from keras.optimizers import SGD
nb_class = 7

def build_model(mode):

    """Return the Keras model for training

    Keyword arguments:
    mode: model name specified in training and predicting script

    """
    model = Sequential()
    if mode == 'easy':
        # CNN part (you can repeat this part several times)
        model.add(Convolution2D(8,1,1,border_mode='valid',input_shape=(48,48,1)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.8))

        # Fully connected part
        model.add(Flatten())
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(nb_class))
        model.add(Activation('softmax'))
        opt = SGD(lr=0.01,decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    model.summary() # show the whole model in terminal
    return model


from keras.callbacks import Callback

class History(Callback):
    def on_train_begin(self,logs={}):
        self.tr_losses=[]
        self.val_losses=[]
        self.tr_accs=[]
        self.val_accs=[]

    def on_epoch_end(self,epoch,logs={}):
        self.tr_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.tr_accs.append(logs.get('acc'))
        self.val_accs.append(logs.get('val_acc'))


