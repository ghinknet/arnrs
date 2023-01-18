import os

from detector import Detector
from paddleocr import PaddleOCR

import cv2
import easyocr
import requests

class number(object):
    def __init__(self, gpu=False, times=2):
        assert type(gpu) is bool
        assert type(times) is int and times >= 1
        self.gpu = gpu
        self.times = times

        # Init detector and OCR
        self.__detector = Detector(device="gpu" if gpu else "cpu")
        self.__eocr = easyocr.Reader(['ch_sim', 'en'], gpu=self.gpu)
        self.__pocr = PaddleOCR(use_angle_cls=True, use_gpu=self.gpu, show_log=False)

    def recognize(self, image):
        '''
        Recognize aircraft registration number with detection
        and ocr powered by pytorch and paddlepaddle engine.
        :image Accept numpy array or image file path
        '''
        if type(image) is str:
            path = os.path.abspath(image)
            image = cv2.imread(path)

        result = self.__detector.image(image)
        # Subjective judgment
        area_max = 0
        area_index = 0
        for i in range(len(result[1])):
            d = result[1][i]
            this_area = ((d["box"][1] - d["box"][0]) ** 2 + (d["box"][3] - d["box"][2]) ** 2) ** 0.5
            if this_area > area_max and result[1][i]["class"] == "airplane":
                area_max = this_area
                area_index = i

        i = result[1][area_index]
        img = image[int(i["box"][1]):int(i["box"][3]), int(i["box"][0]):int(i["box"][2])]

        ocr_result = []
        ocr_filter = []

        for _ in range(2):
            eocr_result = self.__eocr.readtext(img, detail=1)
            pocr_result = self.__pocr.ocr(img, cls=True)

            for e in eocr_result:
                if e[2] > 0.6 and e[1] not in ocr_filter:
                    ocr_result.append(
                        (tuple([tuple(i) for i in e[0]]), e[1], e[2])
                    )
                    ocr_filter.append(e[1])
            for p in pocr_result[0]:
                if p[1][1] > 0.6 and e[1][0] not in ocr_filter:
                    ocr_result.append(
                        (tuple([tuple(i) for i in p[0]]), e[1][0], e[1][1])
                    )
                    ocr_filter.append(e[1][0])

        # Read database
        for i in ocr_result:
            db = requests.post("http://www.airframes.org/", data={"reg1": i[1]}).text
            if "No data found on this query." not in db:
                return i

        return None