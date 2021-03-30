import tensorflow as tf
import numpy as np
import args as cfg
import argparse
import cv2
import os

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from utils.data_aug import letterbox_resize

from time import localtime

from model import yolov3

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

from plate_rec import HyperLPR_plate_recognition

fontC = ImageFont.truetype('platech.ttf', 32, 0)


class detector(object):

    def __init__(self):

        self.new_size = cfg.img_size

        self.resize = True

        self.class_name_path = './data/mydata.names'

        self.restore_path = cfg.restore_path

        self.classes = read_class_names(self.class_name_path)

        self.num_class = len(self.classes)

        self.color_table = get_color_table(self.num_class)

        self.sess = tf.Session()
        self.init_params()

    def get_time(self, name):

        out = '{}年 {}月 {}日\n时间：{}\n识别车牌：{}\n'
        t = localtime()
        out = out.format(
            t.tm_year, t.tm_mon, t.tm_mday,
            str(t.tm_hour)+':'+str(t.tm_min)+':'+str(t.tm_sec),
            name
        )

        return out

    def drawTest(self, image, addText, x1, y1):

        img = Image.fromarray(image)
        draw = ImageDraw.Draw(img)
        draw.text((x1, y1),
                  addText.encode("utf-8").decode("utf-8"),
                  (0, 200, 255), font=fontC)
        imagex = np.array(img)

        return imagex

    def init_params(self):

        self.input_data = tf.placeholder(
            tf.float32, [1, self.new_size[1], self.new_size[0], 3], name='input_data')
        self.yolo_model = yolov3(self.num_class, cfg.anchors)

        with tf.variable_scope('yolov3'):
            self.pred_feature_maps = self.yolo_model.forward(
                self.input_data, False)
        self.pred_boxes, self.pred_confs, self.pred_probs = self.yolo_model.predict(
            self.pred_feature_maps)

        self.pred_scores = self.pred_confs * self.pred_probs

        self.boxes, self.scores, self.labels = gpu_nms(
            self.pred_boxes, self.pred_scores, self.num_class, max_boxes=200, score_thresh=0.25, nms_thresh=0.45)

        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.restore_path)

    def demo(self, pt):
        
        img_ori = cv2.imread(pt)
        if self.resize:
            img, resize_ratio, dw, dh = letterbox_resize(
                img_ori, self. new_size[0], self.new_size[1])
        else:
            height_ori, width_ori = img_ori.shape[:2]
            img = cv2.resize(img_ori, tuple(self.new_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] - 127.5

        boxes_, scores_, labels_ = self.sess.run(
            [self.boxes, self.scores, self.labels], feed_dict={self.input_data: img})

        # rescale the coordinates to the original image
        if letterbox_resize:
            boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
            boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
        else:
            boxes_[:, [0, 2]] *= (width_ori/float(new_size[0]))
            boxes_[:, [1, 3]] *= (height_ori/float(new_size[1]))

        for i in range(len(boxes_)):
            x0, y0, x1, y1 = boxes_[i].astype(np.int)
            cv2.rectangle(img_ori, (x0, y0), (x1, y1), (0, 200, 255), 4)
            res, con = HyperLPR_plate_recognition(img_ori, (x0, y0, x1, y1))
            label = '置信度: {:.2f}%\n'.format(
                scores_[i] * 100) + self.get_time(res)
            img_ori = self.drawTest(img_ori, label, 10, 10)

        cv2.imshow('result', img_ori)
        cv2.waitKey(0)

    def detect(self, img_ori):

        if self.resize:
            img, resize_ratio, dw, dh = letterbox_resize(
                img_ori, self. new_size[0], self.new_size[1])
        else:
            height_ori, width_ori = img_ori.shape[:2]
            img = cv2.resize(img_ori, tuple(self.new_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] - 127.5

        boxes_, scores_, labels_ = self.sess.run(
            [self.boxes, self.scores, self.labels], feed_dict={self.input_data: img})

        # rescale the coordinates to the original image
        if letterbox_resize:
            boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
            boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
        else:
            boxes_[:, [0, 2]] *= (width_ori/float(new_size[0]))
            boxes_[:, [1, 3]] *= (height_ori/float(new_size[1]))

        for i in range(len(boxes_)):
            x0, y0, x1, y1 = boxes_[i].astype(np.int)
            cv2.rectangle(img_ori, (x0, y0), (x1, y1), (0, 200, 255), 4)
            res, con = HyperLPR_plate_recognition(img_ori, (x0, y0, x1, y1))
            label = '置信度: {:.2f}%\n'.format(
                scores_[i] * 100) + self.get_time(res)
            img_ori = self.drawTest(img_ori, label, 10, 10)

        return img_ori


if __name__ == '__main__':

    det = detector()

    test_path = './demo_images'

    im_names = os.listdir(test_path)

    for name in im_names:

        pt = os.path.join(test_path, name)

        det.demo(pt)

    cv2.destroyAllWindows()
