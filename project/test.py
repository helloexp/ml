import numpy as np
import cv2
import os

np.random.seed(0)

# test different containers
for fn_mask in ['test_%s.mkv', 'test_%s.avi', 'test_%s.wmv']:
  # test different codecs
  for fourcc_name, fourcc in [('uncompressed', 0), ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')), ('xvid', cv2.VideoWriter_fourcc(*'MJPG'))]:
    fn = fn_mask % fourcc_name

    writer = cv2.VideoWriter(fn, fourcc, 25, (100, 100))

    # generate random 100x100 images
    for i in range(0,100):
        img = np.random.random_integers(0, 255, (100,100,3)).astype(np.uint8)
        writer.write(img)

    writer.release()

    # test result (is generated file too small?)
    fs = os.path.getsize(fn)

    print('%s [%dB] [%s]' % (fn, fs, 'FAIL' if fs<10000 else 'OK'))