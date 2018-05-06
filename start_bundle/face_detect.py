# -*- coding:utf-8 -*-


import numpy as np
import argparse
import dlib
import cv2


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def shape_to_np(shape,dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords


def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)


def detect_face_detail(img_name):
    img = cv2.imread(img_name)

    # img = cv2.resize(img, (520, 500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    racts = detector(gray, 1)

    for (i,ract) in enumerate(racts):
        shape=predictor(gray,ract)

        coords = shape_to_np(shape)

        (x, y, w, h) = rect_to_bb(ract)

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(img, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for (x, y) in coords:
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow("Output", img)
    cv2.waitKey(0)


if __name__ == '__main__':

    detect_face_detail("../resource/IMG_2030.jpg")




