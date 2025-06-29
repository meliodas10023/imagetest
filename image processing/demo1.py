import cv2
import numpy as np
import matplotlib.pyplot as plt

def gamma_correction(image, gamma=1.0):
    # 归一化图像到 [0, 1] 范围
    image_normalized = image / 255.0
    # 应用伽马校正
    corrected_image = np.power(image_normalized, gamma)
    # 将图像值还原到 [0, 255] 范围
    corrected_image = np.uint8(corrected_image * 255)
    return corrected_image

if __name__ == '__main__':
    # 读取灰度图像
    image = cv2.imread('images/7-复杂场景下车牌识别--昏暗场景---该图片仅用于教学实验，务必不能上网或者随意传播，违者必究.png', 0)
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)
    hist = cv2.calcHist([image], [0], None, [256], [0, 255])
    # plt.subplots((2.2) ,size = (8.6))
    plt.plot(hist)
    plt.show()

    # 应用伽马校正（例如 gamma=0.5 使图像变亮，gamma=1.5 使图像变暗）
    gamma = 0.5  # 调整 gamma 值
    corrected_image = gamma_correction(image, gamma)
    cv2.imshow(f"Gamma Corrected (gamma={gamma})", corrected_image)
    cv2.waitKey(0)
    hist = cv2.calcHist([corrected_image], [0], None, [256], [0, 255])
    # plt.subplots((2.2) ,size = (8.6))
    plt.plot(hist)
    plt.show()

    cv2.destroyAllWindows()
