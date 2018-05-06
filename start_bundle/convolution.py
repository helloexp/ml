# -*- coding:utf-8 -*-

from skimage.exposure import rescale_intensity
import numpy as np
import cv2


def convolue(image, K):
    (iH, iW) = image.shape[:2]
    (kH, kW) = K.shape[:2]

    pad = (kW - 1) // 2
    print(pad)

    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float")
    print(image.shape)

    for y in range(pad, pad + iH):
        for x in range(pad, iW + pad):
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            k = (roi * K).sum()
            output[y - pad, x - pad] = k

    output = rescale_intensity(output, (0, 255))
    output = (output * 255).astype("uint8")
    # cv2.imshow("win", output)
    # cv2.waitKey(0)
    return output


if __name__ == '__main__':
    image = cv2.imread("../resource/two_cat.jpg", cv2.IMREAD_GRAYSCALE)
    smallBlur = np.ones((7, 7)) * (1 / (7 * 7))
    largeBlur = np.ones((21, 21)) * (1 / (21 * 21))

    sharpen = np.asarray([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype="int")

    laplacian = np.array((
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]), dtype="int")

    sobelX = np.array((
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]), dtype="int")

    sobelY = np.array((
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]), dtype="int")

    emboss = np.array((
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]), dtype="int")

    kernelBank = (
        ("small_blur", smallBlur),
        ("large_blur", largeBlur),
        ("sharpen", sharpen),
        ("laplacian", laplacian),
        ("sobel_x", sobelX),
        ("sobel_y", sobelY),
        ("emboss", emboss))

    for (kernelName,K) in kernelBank:

        print("[INFO] applying {} kernel".format(kernelName))
        convolveOutput = convolue(image, K)
        opencvOutput = cv2.filter2D(image,-1,K)

        cv2.imshow("Original", image)
        cv2.imshow("{} - convole".format(kernelName), convolveOutput)
        cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

