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
    def __init__(self, gpu=False, times=1, filter=0.8, ua="ARNRS", debug=False):
        assert type(gpu) is bool
        assert type(times) is int and times >= 1
        self.gpu = gpu
        self.times = times
        self.similar = {"8": "B", "O": "0", "-": "—", "1": "/", "l": "I", "2": "Z", "4": "A"}
        self.ua = ua
        self.filter = filter
        self.debug = debug

        # Init detector and OCR
        self.__detector = Detector(device="cuda" if gpu else "cpu")
        self.__eocr = easyocr.Reader(['ch_sim', 'en'], gpu=self.gpu)
        self.__pocr = PaddleOCR(use_angle_cls=True, use_gpu=self.gpu, show_log=debug)

        # Init database
        self.__database = {}
        i = 0
        with open('aircraftDatabase.csv', "r", encoding='utf-8') as fb:
            for row in csv.reader(fb, skipinitialspace=True):
                if not i:
                    keys = row
                else:
                    self.__database[row[1]] = dict(zip(keys, row))
                    self.__database[row[1].replace("-", "")] = dict(zip(keys, row))
                i += 1

        # Try to update database
        '''
        update_database_daemon_thread = threading.Thread(target=self.__update_database_daemon, name="Update Database Daemon Thread")
        update_database_daemon_thread.daemon = True
        update_database_daemon_thread.start()
        '''

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

    def __distance(self, p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def __closest_pair(self, X, Y):
        if len(X) <= 3:
            return min([self.__distance(X[i], X[j]) for i in range(len(X)) for j in range(i + 1, len(X))])

        mid = len(X) // 2
        XL, XR = X[:mid], X[mid:]
        YL, YR = [p for p in Y if p in XL], [p for p in Y if p in XR]

        d = min(self.__closest_pair(XL, YL), self.__closest_pair(XR, YR))

        line = (X[mid][0] + X[mid-1][0]) / 2
        YS = [p for p in Y if abs(p[0] - line) < d]

        return min(d, self.__closest_split_pair(YS, d))

    def __closest_split_pair(self, Y, d):
        n = len(Y)
        for i in range(n - 1):
            for j in range(i + 1, min(i + 8, n)):
                if self.__distance(Y[i], Y[j]) < d:
                    d = self.__distance(Y[i], Y[j])
        return d

    def __dis(self, p1, p2):
        X = p1 + p2
        Y = sorted(X, key=lambda p: (p[0], p[1]))
        return self.__closest_pair(X, Y)

    def search(self, keyword):
        '''
        Search a plane by it registration number
        :keyword Registration number
        '''
        if keyword.upper() in self.__database.keys() and not keyword.isdigit():
            return self.__database[keyword.upper()]
        
        # Similar characters replace
        self.similar = {**self.similar, **dict(zip(self.similar.values(), self.similar.keys()))}
        condition = []
        for i in range(1, len(self.similar.items()) + 1):
            condition.extend(list(itertools.combinations(self.similar.items(), i)))

        for c in condition:
            for c_i in c:
                keyword_temp = keyword.replace(c_i[0], c_i[1])
                if keyword_temp.upper() in self.__database.keys():
                    return self.__database[keyword_temp.upper()]
        
        if keyword.upper() in self.__database.keys() and keyword.isdigit():
            return self.__database[keyword.upper()]

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

        for _ in range(self.times):
            # OCR recognize
            pocr_result = self.__pocr.ocr(img, cls=True)
            if self.debug:
                print(pocr_result)
                print("------------------------------B-")
            eocr_result = self.__eocr.readtext(img, detail=1)
            if self.debug:
                print(eocr_result)
                print("------------------------------A-")

            # OCR results tidy up
            for i in range(len(pocr_result[0])):
                for j in range(len(pocr_result[0])):
                    if self.debug:
                        print("------------------------------D-")
                        print(pocr_result[0][i][0], pocr_result[0][j][0])
                    if i != j and len(pocr_result[0][i][0]) == 4 and len(pocr_result[0][j][0]) == 4 and self.__dis(pocr_result[0][i][0], pocr_result[0][j][0]) < 5:
                        if self.debug:
                            print("D Appended")
                        pocr_result.append(((pocr_result[0][i][0], pocr_result[0][j][0]), pocr_result[0][i][1][1] + pocr_result[0][j][1][1], (pocr_result[0][i][1][2] + pocr_result[0][j][1][2]) / 2))
                        pocr_result.append(((pocr_result[0][j][0], pocr_result[0][i][0]), pocr_result[0][j][1][1] + pocr_result[0][i][1][1], (pocr_result[0][j][1][2] + pocr_result[0][i][1][2]) / 2))
                    else:
                        if self.debug:
                            disout = 0
                            if len(eocr_result[i][0]) == 4 and len(eocr_result[j][0]) == 4:
                                disout = self.__dis(pocr_result[0][i][0], pocr_result[0][j][0])
                            print(i != j, len(pocr_result[0][i][0]) == 4, len(pocr_result[0][j][0]) == 4, disout)
                            print("------------------------------D-")

            for i in range(len(eocr_result)):
                for j in range(len(eocr_result)):
                    if self.debug:
                        print("------------------------------C-")
                        print(eocr_result[i][0], eocr_result[j][0])
                    if i != j and len(eocr_result[i][0]) == 4 and len(eocr_result[j][0]) == 4 and self.__dis(eocr_result[i][0], eocr_result[j][0]) < 5:
                        if self.debug:
                            print("C Appended")
                        eocr_result.append(((eocr_result[i][0], eocr_result[j][0]), eocr_result[i][1] + eocr_result[j][1], (eocr_result[i][2] + eocr_result[j][2]) / 2))
                        eocr_result.append(((eocr_result[j][0], eocr_result[i][0]), eocr_result[j][1] + eocr_result[i][1], (eocr_result[j][2] + eocr_result[i][2]) / 2))
                    else:
                        if self.debug:
                            disout = 0
                            if len(eocr_result[i][0]) == 4 and len(eocr_result[j][0]) == 4:
                                disout = self.__dis(eocr_result[i][0], eocr_result[j][0])
                            print(i != j, len(eocr_result[i][0]) == 4, len(eocr_result[j][0]) == 4, disout)
                            print("------------------------------C-")

            if self.debug:
                print(pocr_result)
                print(eocr_result)

            # OCR results sum up
            for p in pocr_result[0]:
                if p[1][1] > self.filter and p[1][0] not in ocr_filter:
                    ocr_result.append(
                        (tuple([tuple(i) for i in p[0]]), p[1][0], p[1][1])
                    )
                    ocr_filter.append(p[1][0])
            for e in eocr_result:
                if e[2] > self.filter and e[1] not in ocr_filter:
                    ocr_result.append(
                        (tuple([tuple(i) for i in e[0]]), e[1], e[2])
                    )
                    ocr_filter.append(e[1])

        ocr_result = sorted(ocr_result, key=lambda x:len(x[1]), reverse=True)

        # Read database
        for i in ocr_result:
            r = self.search(i[1])
            if r:
                return r
        
        if self.debug:
            print(ocr_result)
        return None

if __name__ == "__main__":
    import json
    num = number()
    os.makedirs("out", exist_ok=True)
    for pic in os.listdir("test"):
        with open(os.path.join("out", f"{pic}.json"), "w+") as fb:
            fb.write(json.dumps(num.recognize(os.path.join("test", pic))))