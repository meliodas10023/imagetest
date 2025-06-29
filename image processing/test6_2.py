import cv2
import numpy as np

def preprocess_image(image):
    # 灰度化
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 直方图均衡化
    equalized_image = cv2.equalizeHist(gray_image)
    # 高斯滤波去噪
    blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)
    return blurred_image

def locate_license_plate(image):
    # Canny边缘检测
    edges = cv2.Canny(image, 50, 150)
    # 形态学闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # 查找轮廓
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 筛选尺寸合适的候选区域
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 50 < w < 300 and 15 < h < 100:  # 根据车牌尺寸调整参数
            return image[y:y+h, x:x+w]  # 裁剪车牌区域
    return None

def binarize_plate(image):
    # 灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary
#
# def segment_characters(image):
#     # 垂直投影
#     vertical_projection = np.sum(image, axis=0)
#     # 根据投影分割字符
#     characters = []
#     start = 0
#     for i in range(1, len(vertical_projection)):
#         if vertical_projection[i] == 0 and vertical_projection[i-1] != 0:
#             characters.append(image[:, start:i])
#             start = i
#     return characters
#
# def recognize_characters(characters):
#     # 使用模板匹配进行字符识别
#     recognized_text = ""
#     for char in characters:
#         # 这里可以添加模板匹配的逻辑
#         recognized_text += "X"  # 示例字符
#     return recognized_text

if __name__ == '__main__':
    # 读取图像
    image = cv2.imread('images/ps1.png')
    cv2.resize(image, (640, 480))
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)
    # 预处理图像
    preprocessed_image = preprocess_image(image)
    cv2.imshow("Preprocessed Image", preprocessed_image)
    cv2.waitKey(0)
    # 定位车牌
    plate_region = locate_license_plate(preprocessed_image)
    if plate_region is not None:
        cv2.imshow("Plate Region", plate_region)
        cv2.waitKey(0)
        # 车牌图像预处理
        binary_plate = binarize_plate(plate_region)
        cv2.imshow("Binary Plate", binary_plate)
        cv2.waitKey(0)
    #     # 字符分割
    #     characters = segment_characters(binary_plate)
    #     for i, char in enumerate(characters):
    #         cv2.imshow(f"Character {i+1}", char)
    #         cv2.waitKey(0)
    #     # 字符识别
    #     recognized_text = recognize_characters(characters)
    #     print("Recognized Text:", recognized_text)
    # cv2.destroyAllWindows()
