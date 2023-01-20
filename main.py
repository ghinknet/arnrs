import os
import threading
import csv
import time
import itertools

from detector import Detector
from paddleocr import PaddleOCR

import cv2
import easyocr
import requests

class number(object):
    def __init__(self, gpu=False, times=2, ua="ARNRS"):
        assert type(gpu) is bool
        assert type(times) is int and times >= 1
        self.gpu = gpu
        self.times = times
        self.similar = {"8": "B", "o": "0", "-": "—"}
        self.ua = ua

        # Init detector and OCR
        self.__detector = Detector(device="gpu" if gpu else "cpu")
        self.__eocr = easyocr.Reader(['ch_sim', 'en'], gpu=self.gpu)
        self.__pocr = PaddleOCR(use_angle_cls=True, use_gpu=self.gpu, show_log=False)

        # Init database
        self.__database = []
        i = 0
        with open('aircraftDatabase.csv', "r", encoding='utf-8') as fb:
            for row in csv.reader(fb, skipinitialspace=True):
                if not i:
                    keys = row
                else:
                    self.__database.append(dict(zip(keys, row)))
                i += 1

        # Try to update database
        update_database_daemon_thread = threading.Thread(target=self.__update_database_daemon, name="Update Database Daemon Thread")
        update_database_daemon_thread.daemon = True
        update_database_daemon_thread.start()

    def __update_database_daemon(self):
        while True:
            update_database_thread = threading.Thread(target=self.__update_database, name="Update Database Thread")
            update_database_thread.daemon = True
            update_database_thread.start()

            time.sleep(60 * 60 * 1)

    def __update_database(self):
        f = 0
        while True:
            try:
                database = requests.get("https://opensky-network.org/datasets/metadata/aircraftDatabase.csv", headers={"user-agent": self.ua}).text
            except Exception as e:
                print("Failed to update local registration number database,", e, ", retrying... Times: ", f+1)
                f += 1
                if f >= 10:
                    break
            else:
                with open('aircraftDatabase.csv', "w+", encoding='utf-8') as fb:
                    fb.write(database)
                self.__database = []
                i = 0
                with open('aircraftDatabase.csv', "r", encoding='utf-8') as fb:
                    for row in csv.reader(fb, skipinitialspace=True):
                        if not i:
                            keys = row
                        else:
                            self.__database.append(dict(zip(keys, row)))
                        i += 1
                break

    def search(self, keyword):
        '''
        Search a plane by it registration number
        :keyword Registration number
        '''
        for i in self.__database:
            if i["registration"] == keyword:
                return i
        
        # Similar characters replace
        self.similar = {**self.similar, **dict(zip(self.similar.values(), self.similar.keys()))}
        condition = []
        for i in range(1, len(self.similar.items()) + 1):
            condition.extend(list(itertools.combinations(self.similar.items(), i)))

        for c in condition:
            for c_i in c:
                keyword_temp = keyword.replace(c_i[0], c_i[1])
                for i in self.__database:
                    if i["registration"] == keyword_temp:
                        return i
        
        return None

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
            r = self.search(i[1])
            if r:
                return r
        
        return None