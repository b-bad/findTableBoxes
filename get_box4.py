import cv2
import os
import numpy as np
import math




def process_single_image(img_path, show=True, save=False, scale=20, erode_iters=1, dilate_iters=2):

    thresh = 0.5
    img = cv2.imread(img_path)
    h, w, c = img.shape
    if h * w < 4000000:
        scale = 10
    dst = np.zeros((h, w, c), dtype=np.uint8)

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
    print(vertical_lines.__len__())
    print(horizontal_lines.__len__())
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
                           (255, 255, 255), 2)

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
                               (255, 255, 255), 2)
    '''for result_col in results_col:
        for i in range(len(result_col) - 1):
            x1 = int(result_col[i][0])
            y1 = int(result_col[i][1])
            x2 = int(result_col[i + 1][0])
            y2 = int(result_col[i + 1][1])
            for v_line in vertical_lines:
                if min(y1, y2) < v_line[5] < max(y1, y2) and \
                    min(x1, x2) - 10 < v_line[4] < max(x1, x2) + 10:
                    dst = cv2.line(dst, (x1, y1), (x2, y2),
                                   (255, 255, 255), 2)
                    break
    for result_row in results_row:
        for i in range(len(result_row) - 1):
            x1 = int(result_row[i][0])
            y1 = int(result_row[i][1])
            x2 = int(result_row[i + 1][0])
            y2 = int(result_row[i + 1][1])
            for h_line in horizontal_lines:
                if min(x1, x2) < h_line[4] < max(x1, x2) and \
                    min(y1, y2) - 10 < h_line[5] < max(y1, y2) + 10:
                    dst = cv2.line(dst, (x1, y1), (x2, y2),
                                   (255, 255, 255), 2)
                    break'''
    # cv2.imshow("d", dst)
    # cv2.waitKey(0)
    return dst

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
    '''for v in vertical_lines:
        if left_edge - 5 < v[4] < right_edge + 5 and \
                up_edge - 5 < v[5] < low_edge + 5:
            v_lines.append(v)
    for h in horizontal_lines:
        if left_edge - 5 < h[4] < right_edge + 5 and \
                up_edge - 5 < h[5] < low_edge + 5:
            h_lines.append(h)'''

    return vertical_lines, horizontal_lines

def draw_line(img, lines):
    for line in lines:
        img = cv2.line(img, (line[0], line[1]), (line[2], line[3]),
                       (0, 255, 0), 2)

    return img




if __name__ == "__main__":
    path = "C:/Users/Administrator_wzz/Desktop/sx/get-lines-of-table/large"
    file_list = os.listdir(path)
    file_list_ = [path + "/" + file for file in file_list]
    for file in file_list_:
        result = process_single_image(img_path=file, show=False)
        cv2.imwrite("C:/Users/Administrator_wzz/Desktop/sx/get-lines-of-table/large_result/" + file.split('/')[-1],
                    result)














        '''img = cv2.imread(file)
        v_lines, h_lines, vertical_lines, horizontal_lines = get_LSD_result(img)
        LSDresult = draw_line(img, vertical_lines)
        LSDresult = draw_line(LSDresult, horizontal_lines)
        # cv2.imshow("result", LSDresult)
        cv2.imwrite("C:/Users/Administrator_wzz/Desktop/sx/get-lines-of-table/clear_result/"+file.split('/')[-1], LSDresult)
        print(file)'''
