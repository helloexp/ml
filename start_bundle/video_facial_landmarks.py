from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def start_detect():


    # video_stream = VideoStream(usePiCamera=True).start()

    cap = cv2.VideoCapture(0)

    time.sleep(2.0)

    while True:
        
        # get a frame
        ret, frame = cap.read()
        print(ret)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        rects = detector(gray,0)

        for rect in rects:

            shape = predictor(frame, rect)

            shape = imutils.face_utils.shape_to_np(shape)

            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break





if __name__ == '__main__':

    start_detect()



