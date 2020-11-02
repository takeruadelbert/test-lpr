import os

import cv2
import numpy as np

from src.value import *


def crop_bounding_box(img, x, y, x_plus_w, y_plus_h):
    return img[y:y_plus_h, x:x_plus_w]


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


class LicensePlateRecognition:
    weight = os.getcwd() + os.getenv("yolo-weight", "/yolo-obj_final.weights")
    config = os.getcwd() + os.getenv("yolo-config", "/yolo-obj.cfg")
    classPath = os.getcwd() + os.getenv("yolo-class", "/classes.txt")

    def __init__(self, image):
        self.image = image
        self.width = image.shape[1]
        self.height = image.shape[0]
        self.classes = open(self.classPath).read().strip().split("\n")

    def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h, license_plate_number=None):
        if license_plate_number is None:
            label = "{}: {:.4f}".format(str(self.classes[class_id]), confidence)
        else:
            label = "{}: {}".format(str(self.classes[class_id]), license_plate_number)
        color = (213, 255, 0)
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def get_data_from_output_layer(self, outs, class_ids, confidences, boxes):
        conf_threshold = 0.5
        nms_threshold = 0.4
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > CONFIDENCE_LIMIT:
                    center_x = detection[0] * self.width
                    center_y = detection[1] * self.height
                    w = detection[2] * self.width
                    h = detection[3] * self.height
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        return cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    def get_yolo_data_from_bounding_box_image(self, class_id, x, y, w, h):
        detection2 = w / self.width
        detection3 = h / self.height
        center_x = w / 2 + x
        center_y = h / 2 + y
        detection0 = center_x / self.width
        detection1 = center_y / self.height
        return class_id, detection0, detection1, detection2, detection3

    def run(self, filename):
        net = cv2.dnn.readNet(self.weight, self.config)
        blob = cv2.dnn.blobFromImage(self.image, DEFAULT_SCALE, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        indices = self.get_data_from_output_layer(outs, class_ids, confidences, boxes)
        result = []

        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            result.append(self.get_yolo_data_from_bounding_box_image(class_ids[i], x, y, w, h))

        if result:
            with open("output/{}.txt".format(filename), "a+") as f:
                for data in result:
                    f.write(' '.join(str(value) for value in data) + '\n')
