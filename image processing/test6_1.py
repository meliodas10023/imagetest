import cv2
import numpy as np

def preprocess_image(image):
    # 灰度化
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 直方图均衡化
    equalized_image = cv2.equalizeHist(gray_image)
    # 图像高斯滤波去噪
    blur = cv2.GaussianBlur(gray_image, (5, 5), 1, 1)  # 核尺寸通过对图像的调节自行定义
    # 图像阈值化处理
    ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)  # 二进制阈值化


    # 高斯滤波
    blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)

    return equalized_image

# def locate_license_plate(image):
#
#     # Canny边缘检测
#     # 2.Scharr算子
#     # 应用Scharr算子进行边缘检测
#     scharr_x = cv2.Scharr(blur, cv2.CV_32F, 1, 0)  # 水平方向
#     scharr_y = cv2.Scharr(blur, cv2.CV_32F, 0, 1)  # 垂直方向
#
#     # 计算梯度幅值
#     gradient_magnitude = cv2.magnitude(scharr_x, scharr_y)
#     # 将梯度幅值转换为8位图像
#     gradient_magnitude = np.uint8(gradient_magnitude)
#     # 显示结果
#     cv2.imshow("Scharr Edge Detection", gradient_magnitude)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     cv2.imshow('Canny',edges)
#     cv2.waitKey(0)
#
#     # 定义结构元素并进行闭运算
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))
#     closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
#
#     # 查找轮廓
#     contours, _ = cv2.findContours(closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     plate_region = None
#     # 筛选尺寸合适的候选区域
#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         if 50 < w < 300 and 15 < h < 100:  # 根据车牌尺寸调整参数
#             cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             plate_region = blurred_image[y:y+h, x:x+w]  # 裁剪车牌区域
#             break
#
#     return plate_region


if __name__ == '__main__':    # 读取图像
    image = cv2.imread('images/ps1.png')
    cv2.resize(image, (640, 480))
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)
    #  预处理图像
    blurred_image = preprocess_image(image)
    cv2.imshow("Preprocessed Image", blurred_image)
    cv2.waitKey(0)
    # # 定位车牌
    # plate_region = locate_license_plate(blurred_image)
    # cv2.imshow("Plate Region", plate_region)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

