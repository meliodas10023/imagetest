import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取图像并预处理（灰度化 + 去噪）
image = cv2.imread('images/ts.jpg')  # 替换为你的图像路径

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow('Blurred', blurred)
cv2.waitKey(0)


# 2. 二值化处理（使用 Otsu 自动阈值）
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('Binary', binary)
cv2.waitKey(0)


# 3. 边缘检测（Canny）
edges = cv2.Canny(binary, 50, 150)

# 4. 形态学操作（闭运算修复断边 + 膨胀扩大连通性）
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
dilated = cv2.dilate(closed, kernel, iterations=2)

# 5. 找轮廓并填充（获取最终的完整水果区域二值图）
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filled = np.zeros_like(gray)
cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

# 可视化各步骤
plt.figure(figsize=(15, 6))
plt.subplot(1, 5, 1), plt.title("Gray"), plt.imshow(gray, cmap='gray'), plt.axis('off')
plt.subplot(1, 5, 2), plt.title("Binary"), plt.imshow(binary, cmap='gray'), plt.axis('off')
plt.subplot(1, 5, 3), plt.title("Edges"), plt.imshow(edges, cmap='gray'), plt.axis('off')
plt.subplot(1, 5, 4), plt.title("Morph"), plt.imshow(dilated, cmap='gray'), plt.axis('off')
plt.subplot(1, 5, 5), plt.title("Final Filled"), plt.imshow(filled, cmap='gray'), plt.axis('off')
plt.tight_layout()
plt.show()
