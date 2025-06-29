import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def prepprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equ_image = cv2.equalizeHist(gray_image)
    blur_image = cv2.GaussianBlur(equ_image, (3, 3), 0)
    return blur_image


#手动调整HSV阈值
def adjust_hsv_threshold(image):
    def nothing(x):
        pass

    cv2.namedWindow("HSV Adjust")
    cv2.createTrackbar("H Min", "HSV Adjust", 190, 255, nothing)
    cv2.createTrackbar("H Max", "HSV Adjust", 245, 255, nothing)
    cv2.createTrackbar("S Min", "HSV Adjust", int(0.35 * 255), 255, nothing)
    cv2.createTrackbar("S Max", "HSV Adjust", 255, 255, nothing)
    cv2.createTrackbar("V Min", "HSV Adjust", int(0.3 * 255), 255, nothing)
    cv2.createTrackbar("V Max", "HSV Adjust", 255, 255, nothing)

    while True:
        h_min = cv2.getTrackbarPos("H Min", "HSV Adjust")
        h_max = cv2.getTrackbarPos("H Max", "HSV Adjust")
        s_min = cv2.getTrackbarPos("S Min", "HSV Adjust")
        s_max = cv2.getTrackbarPos("S Max", "HSV Adjust")
        v_min = cv2.getTrackbarPos("V Min", "HSV Adjust")
        v_max = cv2.getTrackbarPos("V Max", "HSV Adjust")

        lower_hsv = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper_hsv = np.array([h_max, s_max, v_max], dtype=np.uint8)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return lower_hsv, upper_hsv



def locate_license_plate(image):
    # 转换到HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 定义蓝色车牌的HSV阈值
    lower_blue, upper_blue = adjust_hsv_threshold(image)


    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # cv2.imshow("HSV Mask", mask)
    # cv2.waitKey(0)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 筛选符合尺寸比例的轮廓（车牌近似矩形）
    candidate_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h  # 车牌宽高比一般在2~5之间
        area = cv2.contourArea(cnt)

        # 设置过滤条件：面积和宽高比
        if area > 500 and 2 < aspect_ratio < 5:
            candidate_contours.append((x, y, w, h))

    # 如果没有找到符合条件的轮廓，返回None
    if not candidate_contours:
        return None

    # 按面积排序，取最大一个
    candidate_contours.sort(key=lambda x: x[2] * x[3], reverse=True)
    x, y, w, h = candidate_contours[0]

    # 返回裁剪出的车牌区域
    return image[y:y + h, x:x + w]

def binarize_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary



if __name__ == '__main__':
    image = cv2.imread('images/7-复杂场景下车牌识别-昏暗+污染---该图片仅用于教学实验，务必不能上网或者随意传播，违者必究.png')
    image = cv2.resize(image, (400,300))


    cv2.imshow('Original Image', image)
    cv2.waitKey(0)

    blur_image = prepprocess_image(image)
    cv2.imshow('Preprocessed Image', blur_image)
    cv2.waitKey(0)

    plate_region = locate_license_plate(image)
    cv2.imshow('Detected License Plate', plate_region)
    cv2.waitKey(0)

    binary_plate = binarize_plate(plate_region)
    cv2.imshow('Binarized Plate', binary_plate)
    cv2.waitKey(0)



