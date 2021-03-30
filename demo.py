import cv2
from detector import detector
import numpy as np

if __name__ == '__main__':

    det = detector()

    cap = cv2.VideoCapture('./a.mp4')

    video_width = int(cap.get(3))
    video_height = int(cap.get(4))
    fps = int(cap.get(5))
    # fps = 15
    print(fps)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') #opencv3.0
    videoWriter = cv2.VideoWriter(
        'detected.mp4', fourcc, fps, (video_width, video_height))

    while True:

        _, im = cap.read()
        if im is None:
            break

        raw = im.copy()
        result = det.detect(im)

        cv2.imshow('a', result)
        videoWriter.write(result)
        cv2.waitKey(1)

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()
