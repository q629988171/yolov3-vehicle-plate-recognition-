from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import qtawesome

from PyQt5.Qt import QMutex

from time import sleep, ctime
import numpy as np
import sys
import cv2
import imutils
from .out_GUI_init_layout import Initor_for_event
from detector import detector


class MainUi(Initor_for_event):
    '''
    交通监控功能界面
    '''

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.detector = detector()
        self.setWindowTitle("基于YOLOv3的车牌识别系统")
        self.timer = QTimer(self) # 计时器，用于刷新界面
        self.timer.timeout.connect(self.update)
        self.timer.start(50)
        self.resize(1200, 910)
        self.camera_id = 0
        self.video_size = (500, 360)
        self.detectFlag = 0  # 初始不显示检测结果
        self.playFlag = 0  # 初始不显示摄像头画面
        # self.setFixedSize(self.width(), self.height())
        self.init_layout()
        self.main_layout.setSpacing(0)
        self.init_thread_params()
        self.playvideo = False
        self.video = None
        # 传参 lambda: self.btnstate(self.checkBox2)

    def init_thread_params(self):

        self.init_clik()

    def playon(self):
        self.playvideo = not self.playvideo
        self.video = cv2.VideoCapture(0)


    def init_clik(self):

        self.left_close.clicked.connect(self.close_all)
        self.left_button_2.clicked.connect(self.playon)

    def close_all(self):

        self.close()

    def init_play_btn(self):
        pass

    def init_layout(self):

        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
        self.init_left()
        self.init_right()
        self.init_bottom_box()
        self.init_btn_event()
        self.setWindowOpacity(0.9)  # 设置窗口透明度
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # 设置窗口背景透明
        self.main_layout.setSpacing(0)

    def load_local_video_file(self):

        videoName, _ = QFileDialog.getOpenFileName(
            self, "Open", "", "*.jpg;;*.png;;All Files(*)")
        if videoName != "":  # 为用户取消
            im_in = cv2.imdecode(np.fromfile(videoName,dtype=np.uint8),-1)
            im_in = imutils.resize(im_in, width=500)
            frame = cv2.cvtColor(im_in, cv2.COLOR_BGR2RGB)
            height, width, bytesPerComponent = frame.shape
            bytesPerLine = bytesPerComponent * width
            q_image = QImage(frame.data,  width, height, bytesPerLine,
                             QImage.Format_RGB888).scaled(self.raw_video.width(), self.raw_video.height())
            self.raw_video.setPixmap(QPixmap.fromImage(q_image))
            
            im_out = self.detector.detect(im_in)
            detected_frame = cv2.cvtColor(im_out, cv2.COLOR_BGR2RGB)
            height, width, bytesPerComponent = detected_frame.shape
            bytesPerLine = bytesPerComponent * width
            q_image = QImage(detected_frame.data,  width, height, bytesPerLine,
                                 QImage.Format_RGB888).scaled(self.mask_video.width(), self.mask_video.height())
            self.mask_video.setPixmap(QPixmap.fromImage(q_image))

    def update(self):
    
        if self.playvideo and self.video:  # 为用户取消
            sucess, im_in = self.video.read()
            if sucess:
                im_in = imutils.resize(im_in, width=500)
                frame = cv2.cvtColor(im_in, cv2.COLOR_BGR2RGB)
                height, width, bytesPerComponent = frame.shape
                bytesPerLine = bytesPerComponent * width
                q_image = QImage(frame.data,  width, height, bytesPerLine,
                                QImage.Format_RGB888).scaled(self.raw_video.width(), self.raw_video.height())
                self.raw_video.setPixmap(QPixmap.fromImage(q_image))
                
                im_out = self.detector.detect(im_in)
                detected_frame = cv2.cvtColor(im_out, cv2.COLOR_BGR2RGB)
                height, width, bytesPerComponent = detected_frame.shape
                bytesPerLine = bytesPerComponent * width
                q_image = QImage(detected_frame.data,  width, height, bytesPerLine,
                                    QImage.Format_RGB888).scaled(self.mask_video.width(), self.mask_video.height())
                self.mask_video.setPixmap(QPixmap.fromImage(q_image))

