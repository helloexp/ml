# encoding:utf-8

from PIL import Image
from pylab import *

class ImageOperation(object):

    def __init__(self,size):
        self.size=size

    def array_2_image(self,data):

        image = Image.fromarray(data)
        return image


    def image_2_array(self,filename):

        image = Image.open(filename)

        return array(image)


if __name__ == '__main__':

    operation = ImageOperation((128,128))

    array = operation.image_2_array("../resource/cat.jpeg")

    print(array)

    image = operation.array_2_image(array)

    image.show()
    show()