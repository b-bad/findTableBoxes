import cv2
import numpy as np
import math
from sklearn.cluster import DBSCAN


def dict_sort(dict_, dict_length,axis=0):

    if axis:
        for i in range(dict_length):
            list_ = dict_[str(i)]
            dict_[str(i)] = sorted(list_, key=lambda x: x[0])
    else:
        for i in range(dict_length):
            list_ = dict_[str(i)]
            dict_[str(i)] = sorted(list_, key=lambda x: x[1])

    return dict_


path = 'test_.jpg'
img = cv2.imread(path)
dst = np.ones((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
dst[:, :, :] = 255
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, width = gray.shape

a = 10  # 灰度值均匀化提高对比度
for h in range(height):
    for w in range(width):
        gray[h][w] = 255 - (255 - gray[h][w]) * a
        if gray[h][w] < 0:
            gray[h][w] = 0

LSD = cv2.createLineSegmentDetector(0)
lines = LSD.detect(gray)[0]

PoI = []  # Point of Intersection
temp = []
horizontal_lines = []
vertical_lines = []
mid_of_lines = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    if abs(x1 - x2) <= 5:
        vertical_lines.append(line[0])
    elif abs(y1 - y2) <= 5:
        horizontal_lines.append(line[0])
    mid_of_lines.append([(x1 + x2) / 2, (y1 + y2) / 2])
for horizontal_line in horizontal_lines:
    x11, y11, x12, y12 = horizontal_line
    for vertical_line in vertical_lines:
        x21, y21, x22, y22 = vertical_line
        dis1 = math.sqrt((x11 - x21) ** 2 + (y11 - y21) ** 2)
        dis2 = math.sqrt((x11 - x22) ** 2 + (y11 - y22) ** 2)
        dis3 = math.sqrt((x12 - x21) ** 2 + (y12 - y21) ** 2)
        dis4 = math.sqrt((x12 - x22) ** 2 + (y12 - y22) ** 2)
        if dis1 <= 10:
            temp.append([x11, y11])
            temp.append([x21, y21])
        if dis2 <= 10:
            temp.append([x11, y11])
            temp.append([x22, y22])
        if dis3 <= 10:
            temp.append([x12, y12])
            temp.append([x21, y21])
        if dis4 <= 10:
            temp.append([x12, y12])
            temp.append([x22, y22])
for point in temp:
    if point not in PoI:
        PoI.append(point)  # 去除重复项

db = DBSCAN(eps=10, min_samples=4).fit(PoI)  # 聚类
labels = db.labels_
label_num = max(labels) + 1
PoI_fin = []
for i in range(label_num):
    temp_x = []
    temp_y = []
    for j in range(len(labels)):
        if labels[j] == i:
            temp_x.append(PoI[j][0])
            temp_y.append(PoI[j][1])
    PoI_fin.append([np.median(temp_x), np.median(temp_y)])

PoI_fin_array = np.array(PoI_fin)
PoI_fin_y = np.zeros((len(PoI_fin), 2))
PoI_fin_x = np.zeros((len(PoI_fin), 2))
PoI_fin_y[:, 1] = PoI_fin_array[:, 1]
PoI_fin_x[:, 0] = PoI_fin_array[:, 0]
db_y = DBSCAN(eps=5, min_samples=2).fit(PoI_fin_y)
db_x = DBSCAN(eps=5, min_samples=2).fit(PoI_fin_x)
labels_y = db_y.labels_
labels_x = db_x.labels_
y_num = max(labels_y) + 1
x_num = max(labels_x) + 1

point_classify = {}
for i in range(y_num):
    point_classify[str(i)] = []
for i in range(len(labels_y)):
    point_classify[str(labels_y[i])].append(PoI_fin[i])
point_classify = dict_sort(point_classify, y_num, axis=1)
for i in range(y_num):
    point_list = point_classify[str(i)]
    for j in range(0, len(point_list) - 1):
        x1, y1 = point_list[j]
        x2, y2 = point_list[j + 1]
        for mid_of_line in mid_of_lines:
            x_mid, y_mid = mid_of_line
            if x1 < x_mid < x2 and abs(y_mid - y1) <= 10 and abs(y_mid - y2) <= 10:
                dst = cv2.line(dst, (x1, y1), (x2, y2), (0, 0, 0), thickness=2)
point_classify = {}
for i in range(x_num):
    point_classify[str(i)] = []
for i in range(len(labels_x)):
    if labels_x[i] != -1:
        point_classify[str(labels_x[i])].append(PoI_fin[i])
point_classify = dict_sort(point_classify, x_num, axis=0)
for i in range(x_num):
    point_list = point_classify[str(i)]
    for j in range(0, len(point_list) - 1):
        x1, y1 = point_list[j]
        x2, y2 = point_list[j + 1]
        for mid_of_line in mid_of_lines:
            x_mid, y_mid = mid_of_line
            if y1 < y_mid < y2 and abs(x_mid - x1) <= 10 and abs(x_mid - x2) <= 10:
                dst = cv2.line(dst, (x1, y1), (x2, y2), (0, 0, 0), thickness=2)

dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(dst_gray, 220, 255, cv2.THRESH_BINARY)
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
cv2.imwrite('result.jpg', result)
cv2.waitKey(0)
