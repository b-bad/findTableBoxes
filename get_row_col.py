import cv2
import matplotlib.pyplot as plt
import numpy as np


def naive_nms(boxes_, threshold):

    areas = np.zeros(shape=(len(boxes_)))
    i = 0
    for box_ in boxes_:
        areas[i] = box_[2] * box_[3]
        i += 1
    boxes_ = np.insert(boxes_, 4, values=areas, axis=1)
    boxes_ = boxes_[boxes_[:, 4].argsort()]
    results = []
    i = boxes_[-1]
    boxes_ = np.delete(boxes_, -1, axis=0)
    results.append(i)

    while len(boxes_):
        flag = True
        j = boxes_[-1]
        x11 = j[0]
        y11 = j[1]
        x12 = j[0] + j[2]
        y12 = j[1] + j[3]
        area1 = j[2] * j[3]
        for result in results:
            x21 = result[0]
            y21 = result[1]
            x22 = result[0] + result[2]
            y22 = result[1] + result[3]
            minx = max(x11, x21)
            miny = max(y11, y21)
            maxx = min(x12, x22)
            maxy = min(y12, y22)
            if minx > maxx or miny > maxy:
                # j框与已有矩形框不相交
                continue
            elif x11 >= x21 and x12 <= x22 and y11 >= y21 and y22 >= y12:
                # j框在已有矩形框之内
                flag = False
                break
            else:
                area3 = (maxx - minx) * (maxy - miny)
                if area3/area1 > threshold:
                    # 重叠面积大于设定阈值
                    flag = False
                    break
                else:
                    continue
        if flag:
            results.append(j)
        boxes_ = np.delete(boxes_, -1, axis=0)
        print(len(results))
    return results


def seg_kmeans_gray(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    height = gray.shape[0]
    width = gray.shape[1]

    # 展平
    img_flat = gray.reshape((gray.shape[0] * gray.shape[1], 1))
    img_flat = np.float32(img_flat)

    # 迭代参数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 20, 0.5)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # 聚类
    n = 2
    compactness, labels, centers = cv2.kmeans(img_flat, n, None, criteria, 10, flags)

    # 显示结果

    o1 = np.zeros((gray.shape[0], gray.shape[1], 3))
    o2 = np.zeros((gray.shape[0], gray.shape[1]))

    for i in range(len(labels)):
        if labels[i] == 1:
            o1[i // width][i % width][0] = img[i // width][i % width][0]
            o1[i // width][i % width][1] = img[i // width][i % width][1]
            o1[i // width][i % width][2] = img[i // width][i % width][2]
            o2[i // width][i % width] = 255

    o1 = np.uint8(o1)

    return o1, o2


'''img = cv2.imread('test_b.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = np.ones((3, 3), np.uint8)
erosion = cv2.erode(gray, kernel, iterations=3)
cv2.imshow('e', erosion)
cv2.waitKey(0)'''
path = 'test_b.jpg'
img = cv2.imread(path)
img_c = img.copy()
seg, binary = seg_kmeans_gray(path)
gray = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
kernel = np.ones((3, 3), np.uint8)
img_open = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2)
mser = cv2.MSER_create(_min_area=300)

regions, boxes = mser.detectRegions(img_open)
boxes = naive_nms(boxes, 0.5)
for box in boxes:
    x, y, w, h, area = box
    cv2.rectangle(img_c, (x, y), (x+w, y+h), (0, 255, 0), 2)
print(len(boxes))
plt.subplot(121), plt.imshow(img), plt.title('input')
plt.subplot(122), plt.imshow(img_c, 'gray'), plt.title('output')
plt.show()

