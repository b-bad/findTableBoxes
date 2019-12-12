'''
@:parameter:--dir
            --img
            --save
            --show
            --save_path
'''


import cv2
import os
import numpy as np
import math
import argparse


def get_merge(path):

    img = cv2.imread(path)
    # 二值化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255,  # ~取反，很重要，使二值化后的图片是黑底白字
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.dilate(binary, kernel=kernel, iterations=1)
    #cv2.imshow("binary ", binary)

    rows, cols = binary.shape
    if rows * cols >= 4000000:
        row_scale = 20 # 这个值越大，检测到的直线越多
        col_scale = 20
    else:
        row_scale = 10
        col_scale = 10

    # 识别横线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // col_scale, 1))
    # getStructuringElement： Returns a structuring element of the specified size and shape for morphological operations.
    #  (cols // scale, 1) 为了获取横向的表格线，设置腐蚀和膨胀的操作区域为一个比较大的横向直条
    eroded = cv2.erode(binary, kernel, iterations=1)
    # cv2.imshow("Eroded Image",eroded)
    dilatedcol = cv2.dilate(eroded, kernel, iterations=1)
    # cv2.imshow("Dilated col", dilatedcol)


    # 识别竖线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // row_scale))
    # 竖直方向上线条获取的步骤同上，唯一的区别在于腐蚀膨胀的区域为一个宽为1，高为缩放后的图片高度的一个竖长形直条
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilatedrow = cv2.dilate(eroded, kernel, iterations=1)
    # cv2.imshow("Dilated row", dilatedrow)


    # 标识交点
    bitwiseAnd = cv2.bitwise_and(dilatedcol, dilatedrow)
    #cv2.imshow("bitwiseAnd Image", bitwiseAnd)
    rois = get_rec(bitwiseAnd)
    # print(rois)

    lst = []
    for i, r in enumerate(rois):
        #print(i,r)
        # cv2.imshow("src" + str(i), image[r[3]:r[1], r[2]:r[0]])
        lst.append(list(r))

    #print(lst)
    # 标识表格
    merge = cv2.add(dilatedcol, dilatedrow)
    return merge


def process_single_image(img_path, show=True, save=False, scale=20, erode_iters=1, dilate_iters=2):

    thresh = 0.6
    img = cv2.imread(img_path)
    h, w, c = img.shape
    if h * w < 4000000:
        scale = 10
    dst = np.zeros((h, w), dtype=np.uint8)

    dilate_col, dilate_row = extract_lines(image=img,
                                           scale=scale,
                                           erode_iters=erode_iters,
                                           dilate_iters=dilate_iters,
                                           show=show)
    bitwise_and = get_bit_wise(col=dilate_col, row=dilate_row, show=show)
    rois_list, rois = get_rec(bitwise_and)

    row, col = get_total_row_cols(x_=rois_list)
    row, col = clean_dots(row, col)
    results_row, results_col = get_dots(x=rois_list, row=row, col=col)
    keys_col = results_col.keys()
    keys_row = results_row.keys()
    vertical_lines, horizontal_lines = get_LSD_result(img)

    for v in vertical_lines:
        x1, y1, x2, y2, mid_x, mid_y, length = v
        y_up = min(y1, y2)
        y_low = max(y1, y2)
        y_pt_up = 0
        y_pt_low = img.shape[0] - 1
        x_pt = 0
        flag_y_up = False  # 线在两点间
        flag_y_low = False
        flag_x = False
        for key in keys_row:
            if key <= y_up and key >= y_pt_up:
                y_pt_up = key
                flag_y_up = True
            if key >= y_low and key <= y_pt_low:
                y_pt_low = key
                flag_y_low = True
        for key in keys_col:
            if key - 20 < mid_x < key + 20:
                x_pt = key
                flag_x = True
                break
        if flag_x and flag_y_low and flag_y_up and length >= (y_pt_low - y_pt_up) * thresh:
            dst = cv2.line(dst, (x_pt, y_pt_up), (x_pt, y_pt_low),
                           255, 2)

        for h in horizontal_lines:
            x1, y1, x2, y2, mid_x, mid_y, length = h
            x_left = min(x1, x2)
            x_right = max(x1, x2)
            x_pt_left = 0
            x_pt_right = img.shape[1] - 1
            y_pt = 0
            flag_y = False  # 线在两点间
            flag_x_left = False
            flag_x_right = False
            for key in keys_col:
                if key <= x_left and key >= x_pt_left:
                    x_pt_left = key
                    flag_x_left = True
                if key >= x_right and key <= x_pt_right:
                    x_pt_right = key
                    flag_x_right = True
            for key in keys_row:
                if key - 20 < mid_y < key + 20:
                    y_pt = key
                    flag_y = True
                    break
            if flag_x_right and flag_x_left and flag_y and length >= (x_pt_right - x_pt_left) * thresh:
                dst = cv2.line(dst, (x_pt_left, y_pt), (x_pt_right, y_pt),
                               255, 2)

    # cv2.imshow("d", dst)
    # cv2.waitKey(0)
    return dst

def get_or(img1, img2):
    img3 = cv2.bitwise_or(img1, img2)
    kernel = np.ones((3, 3), np.uint8)
    img3 = cv2.erode(img3, kernel, iterations=1)
    return img3

def get_rec(img_):

    # 在mask那张图上通过findContours 找到轮廓，判断轮廓形状和大小是否为表格。
    _, contours, hierarchy = cv2.findContours(img_, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours_poly = [0] * len(contours)
    boundRect = [0] * len(contours)
    rois = []
    rois_list = []
    for i in range(len(contours)):
        cnt = contours[i]
        # approxPolyDP 函数用来逼近区域成为一个形状，true值表示产生的区域为闭合区域。
        contours_poly[i] = cv2.approxPolyDP(cnt, 2, True)
        # boundingRect为将这片区域转化为矩形，此矩形包含输入的形状。
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        rois.append(np.array(boundRect[i]))
        rois_list.append(list(boundRect[i]))

    return rois_list, rois


def get_total_row_cols(x_):

    row = {}
    col = {}
    num = 1
    for i in range(len(x_) - 1):
        if x_[i][1] == x_[i + 1][1]:
            num += 1
            row[x_[i][1]] = num
        else:
            row[x_[i + 1][1]] = 1
            num = 1
    num = 1
    x_ = sorted(x_, key=lambda x: x[0])
    for i in range(len(x_) - 1):
        if x_[i][0] == x_[i + 1][0]:
            num += 1
            col[x_[i][0]] = num
        else:
            col[x_[i + 1][0]] = 1
            num = 1

    return row, col


def extract_lines(image, scale=20, erode_iters=1, dilate_iters=2, show=True):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY,
                                   15, -10)
    rows, cols = binary.shape

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
    eroded = cv2.erode(binary, kernel, iterations=erode_iters)
    dilated_col = cv2.dilate(eroded, kernel, iterations=dilate_iters)  # 为了是表格闭合，故意使得到的横向更长（以得到交点——bounding-box）

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
    eroded = cv2.erode(binary, kernel, iterations=erode_iters)
    dilated_row = cv2.dilate(eroded, kernel, iterations=dilate_iters)  # 为了是表格闭合，故意使线变长
    if show:
        print("shape:", rows, cols)
        cv2.imshow("Dilated col", dilated_col)
        cv2.imshow("Dilated row", dilated_row)
        # 绘制出横线、竖线
        merge = cv2.add(dilated_col, dilated_row)
        # result_name = "./results_merge/" + img_name  #+ ".jpg"
        # cv2.imwrite(filename=result_name, img=merge)
        cv2.imshow("col & row", merge)

    return dilated_col, dilated_row


def get_bit_wise(col, row, show=True):

    bitwise_and = cv2.bitwise_and(col, row)
    if show:
        cv2.imshow("bitwiseAnd Image", bitwise_and)

    return bitwise_and


def clean_dots(row, col, err=2):

    d = row  # 输入的字典（横坐标：该行点数）
    d_keys = list(d.keys())
    for i in range(len(d_keys) - 1):
        if abs(d_keys[i + 1] - d_keys[i]) <= err:
            d[d_keys[i + 1]] = d[d_keys[i]] + d[d_keys[i + 1]]  # 两点总数合并
            del d[d_keys[i]]  # 删除其中一个
    d2 = col
    d2_keys = list(d2.keys())
    for i in range(len(d2_keys) - 1):
        if abs(d2_keys[i + 1] - d2_keys[i]) <= err:
            d2[d2_keys[i + 1]] = d2[d2_keys[i]] + d2[d2_keys[i + 1]]
            del d2[d2_keys[i]]
    d_keys = list(d.keys())
    d2_keys = list(d2.keys())
    for i in range(len(d_keys)):
        if d[d_keys[i]] <= 1:
            del d[d_keys[i]]
    for i in  range(len(d2_keys)):
        if d2[d2_keys[i]] <= 1:
            del d2[d2_keys[i]]
    return d, d2


def get_dots(x, row, col):
    results_col_ = {}
    results_row_ = {}
    for key in row:
        #     print(row[key])
        #     print("*"*50)
        #     print(key, row[key])
        for val in range(row[key]):
            #         print(key)
            yy = key
            xx = [val[0] for val in x if yy - 5 <= val[1] <= yy + 5]
            result_row_ = [[x, yy] for x in xx]
            result_row_.reverse()
        # print(result)
        results_row_[key] = result_row_  # 原来是append
        # results_row_.reverse()
    for key in col:
        for val in range(col[key]):
            xx = key
            yy = [val[1] for val in x if xx - 5 <= val[0] <= xx + 5]
            result_col_ = [[xx, y] for y in yy]
            result_col_.reverse()
        results_col_[key] = result_col_

    return results_row_, results_col_


def get_LSD_result(img):

    gray = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    binary = cv2.adaptiveThreshold(~gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY,
                                   15, -10)

    LSD = cv2.createLineSegmentDetector(0)
    lines = LSD.detect(binary)[0]

    hough_lines = cv2.HoughLines(binary, 1.0, math.pi / 180, int(min(height, width) / 10))

    horizontal_lines = []
    vertical_lines = []
    mid_of_lines = []
    up_edge = height
    low_edge = 0
    left_edge = width
    right_edge = 0
    max_length = 0
    for line in lines:
        x1, y1, x2, y2 = [int(a) for a in line[0]]

        mid_pt_x = int((x1 + x2) * 0.5)
        mid_pt_y = int((y1 + y2) * 0.5)

        if x1 - x2 == 0:  # 获取水平、竖直直线、表格边界
            vertical_lines.append([x1, y1, x2, y2, mid_pt_x, mid_pt_y, abs(y1 - y2)])
            if abs(y1 - y2) > height / 5 and min(y1, y2) < up_edge:
                up_edge = min(y1, y2)
            if abs(y1 - y2) > height / 5 and max(y1, y2) > low_edge:
                low_edge = max(y1, y2)
            max_length = max(max_length, abs(y1 - y2))
        elif y1 - y2 == 0:
            horizontal_lines.append([x1, y1, x2, y2, mid_pt_x, mid_pt_y, abs(x1 - x2)])
            if abs(x1 - x2) > width / 5 and min(x1, x2) < left_edge:
                left_edge = min(x1, x2)
            if abs(x1 - x2) > width / 5 and max(x1, x2) > right_edge:
                right_edge = max(x1, x2)
            max_length = max(max_length, abs(x1 - x2))
        elif math.atan(abs(x1 - x2) / abs(y1 - y2)) < 5 * math.pi / 180:
            vertical_lines.append([x1, y1, x2, y2, mid_pt_x, mid_pt_y, abs(y1 - y2)])
            if abs(y1 - y2) > height / 5 and min(y1, y2) < up_edge:
                up_edge = min(y1, y2)
            if abs(y1 - y2) > height / 5 and max(y1, y2) > low_edge:
                low_edge = max(y1, y2)
            max_length = max(max_length, abs(y1 - y2))
        elif math.atan(abs(y1 - y2) / abs(x1 - x2)) < 5 * math.pi / 180:
            horizontal_lines.append([x1, y1, x2, y2, mid_pt_x, mid_pt_y, abs(x1 - x2)])
            if abs(x1 - x2) > width / 5 and min(x1, x2) < left_edge:
                left_edge = min(x1, x2)
            if abs(x1 - x2) > width / 5 and max(x1, x2) > right_edge:
                right_edge = max(x1, x2)
            max_length = max(max_length, abs(x1 - x2))

    v_lines = []
    h_lines = []

    return vertical_lines, horizontal_lines

def draw_line(img, lines):
    for line in lines:
        img = cv2.line(img, (line[0], line[1]), (line[2], line[3]),
                       (0, 255, 0), 2)

    return img

def get_bbox_result(img, binary):
    bboxes = []
    height, width = binary.shape
    _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        bboxes.append([x, y, w, h])
        if height * width / 10000 < w * h < height * width / 2 and h >= height / 30 and w >= width /30:
            result = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            result = cv2.putText(result, 'x:' + str(x) + ' y:' + str(y), (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    return result, bboxes

def main(dir, img_p, save_path, save, show):

    if img_p == None or dir == None:
        print("wrong path!")
        exit()
    img_path = dir + "/" + img_p
    if os.path.exists(img_path) == None:
        print("wrong path!")
        exit()
    print("processing " + img_path)
    img = cv2.imread(img_path)
    img1 = process_single_image(img_path=img_path, show=False)
    img2 = get_merge(img_path)
    result = get_or(img1, img2)
    result, bboxs = get_bbox_result(img, result)
    if save:
        if save_path == None:
            save_path = dir
        cv2.imwrite(save_path + "/" + img_p.split(".")[0] + "_result." + img_p.split(".")[-1],
                    result)
        b_box_file = open(save_path + "/" + "bbox.txt", "w+")
        for bbox in bboxs:
            x, y, w, h = bbox
            b_box_file.write(str(x) + " " + str(y) + " "
                             + str(w) + " " + str(h) + "\n")
        b_box_file.close()
    if show:
        cv2.imshow("result", result)
        cv2.waitKey(0)


if __name__ == "__main__":

    parser= argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, help="Dir of the input image")
    parser.add_argument("-i", "--img", type=str)
    parser.add_argument("-sa", "--save", type=bool, default=False, help="Save or not")
    parser.add_argument("-sh", "--show", type=bool, default=False, help="Show or not")
    parser.add_argument("-sp", "--save_path", type=str)
    args = vars(parser.parse_args())
    dir = args["dir"]
    img_p = args["img"]
    save = args["save"]
    show = args["show"]
    save_path = args["save_path"]

    main(dir, img_p, save_path, save, show)
