import cv2
import numpy as np


path = 'test_5.jpg'
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, width = gray.shape
a = 10
for h in range(height):
    for w in range(width):
        gray[h][w] = 255 - (255 - gray[h][w]) * a
        if gray[h][w] < 0:
            gray[h][w] = 0
kernel = np.ones((3, 3), np.uint8)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
blur = cv2.erode(blur, kernel, iterations=3)
blur = cv2.dilate(blur, kernel, iterations=2)
# binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
ret, binary = cv2.threshold(blur, 220, 255, cv2.THRESH_BINARY)
for h in range(height):
    for w in range(width):
        if binary[h][w]:
            binary[h][w] = 0
        else:
            binary[h][w] = 255
scale = 10
horizontal_size = scale
horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
erosion = cv2.erode(binary, horizontal_structure, iterations=2)
dilation = cv2.dilate(erosion, horizontal_structure, iterations=2)

scale2 = 5
horizontal_size2 = scale2
horizontal_structure2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, horizontal_size2))
erosion2 = cv2.erode(binary, horizontal_structure2, iterations=2)
dilation2 = cv2.dilate(erosion2, horizontal_structure2, iterations=2)

mask = dilation + dilation2
joints = cv2.bitwise_and(dilation, dilation2)  # 线交点

_, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if height * width / 400 < w * h < height * width / 3:
        result = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        result = cv2.putText(result, 'x:' + str(x) + ' y:' + str(y), (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

cv2.imshow('result', result)
cv2.imwrite('result_5.jpg', result)
cv2.waitKey(0)
