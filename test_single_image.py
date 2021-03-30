# coding: utf-8

from __future__ import division, print_function

import os
import cv2
from detector import detector

if __name__ == '__main__':

    test_path = './demo_images'
    det = detector()

    im_names = os.listdir(test_path)

    for name in im_names:

        path = os.path.join(test_path, name)
        result = det.detect(cv2.imread(path))
        cv2.imshow('result', result)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
