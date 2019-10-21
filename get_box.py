import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", '--input', type=str, help="Path to the input image")
args = vars(parser.parse_args())
img_path = args["input"]


def func_norm(path):

    boxes = []

    img = cv2.imread(path)
    height, width = img.shape[0], img.shape[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    ret, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(binary, kernel, iterations=3)
    dilation = cv2.dilate(erosion, kernel, iterations=2)
    binary = cv2.medianBlur(dilation, 5)
    for h in range(height):
        for w in range(width):
            if binary[h][w]:
                binary[h][w] = 0
            else:
                binary[h][w] = 255
    canny = cv2.Canny(binary, 50, 150)
    _, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if height * width / 400 < w * h < height * width / 3:
            boxes.append([x, y, h, w])
    return boxes


def func_sp(path):

    boxes = []

    img = cv2.imread(path)
    height, width = img.shape[0], img.shape[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    a = 200
    for h in range(height):
        for w in range(width):
            gray[h][w] = 255 - (255 - gray[h][w]) * a
            if gray[h][w] < 0:
                gray[h][w] = 0
    gray = np.round(gray).astype(np.uint8)
    ret, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(binary, kernel, iterations=3)
    dilation = cv2.dilate(erosion, kernel, iterations=2)
    binary = cv2.medianBlur(dilation, 5)
    for h in range(height):
        for w in range(width):
            if binary[h][w]:
                binary[h][w] = 0
            else:
                binary[h][w] = 255
    canny = cv2.Canny(binary, 50, 150)
    _, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if height * width / 400 < w * h < height * width / 3:
            boxes.append([x, y, h, w])
    return boxes


def detect(path):

    boxes = func_norm(path)
    img = cv2.imread(path)

    if boxes.__len__() < 4:

        boxes = func_sp(path)

    print(boxes)
    for x, y, h, w in boxes:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.imshow("result", img)
    cv2.waitKey(0)


if __name__ == "__main__":

    detect(img_path)


