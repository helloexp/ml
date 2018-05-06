import os

import cv2
import numpy as np

KERNEL_SIZE=3

def estimate_detect_wartermark(foldername):

    images = []
    for r, dirs, files in os.walk(foldername):
        # Get all the images
        for file in files:
            img = cv2.imread(os.sep.join([r, file]))
            if img is not None:
                images.append(img)
            else:
                print("%s not found." % (file))

    # Compute gradients
    # gradx = map(lambda x: cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=KERNEL_SIZE), images)
    # grady = map(lambda x: cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=KERNEL_SIZE), images)
    gradx,grady=compute_gradx(images)

    print(type(gradx))
    print(np.asarray(gradx).shape)
    median_x = np.median(np.asarray(gradx), axis=0)
    median_y = np.median(np.asarray(grady), axis=1)

    print(type(median_x))
    print(type(median_y))

    cv2.imshow("1",median_x)
    cv2.waitKey()

def compute_gradx(images):
    res_x=[]
    res_y=[]
    for img in images:
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=KERNEL_SIZE)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=KERNEL_SIZE)
        res_x.append(sobel_x)
        res_y.append(sobel_y)
    return res_x,res_y

if __name__ == '__main__':

    estimate_detect_wartermark("./mask2")



















