from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers.core import Flatten

class FCHeadNet():

    @staticmethod
    def build(model,classes,D):
        """
        :param model: the body of the network
        :param classes: the total number of classes in our dataset
        :param D: the number of nodes in the fully-connected layer
        :return: 
        """

        headModel = model.output
        headModel=Flatten(name="flatten")(headModel)
        headModel=Dense(D,activation="relu")(headModel)
        headModel=Dropout(rate=0.5)(headModel)
        headModel=Dense(classes,activation="softmax")(headModel)
        return headModel




