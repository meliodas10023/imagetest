import cv2
import numpy as np
import matplotlib.pyplot as plt

# 对原始图像进行灰度化、二值化处理
img_gray = cv2.imread("images/1.jpg", 0)
_, binary_img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)

# 1. 腐蚀
kernel = np.ones((3, 3), np.uint8)
erosion = cv2.erode(binary_img, kernel)

# 2. 膨胀
dilation = cv2.dilate(binary_img, kernel)

# 3. 形态学开运算
opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

# 4. 形态学闭运算
closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

# 使用 matplotlib 展示图像
fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2行3列

# 显示二值化图像
axes[0, 0].imshow(binary_img, cmap='gray')
axes[0, 0].set_title('Binary Image')
axes[0, 0].axis('off')

# 显示腐蚀图像
axes[0, 1].imshow(erosion, cmap='gray')
axes[0, 1].set_title('Eroded Image')
axes[0, 1].axis('off')

# 显示膨胀图像
axes[0, 2].imshow(dilation, cmap='gray')
axes[0, 2].set_title('Dilated Image')
axes[0, 2].axis('off')

# 显示开运算图像
axes[1, 0].imshow(opening, cmap='gray')
axes[1, 0].set_title('Opening Image')
axes[1, 0].axis('off')

# 显示闭运算图像
axes[1, 1].imshow(closing, cmap='gray')
axes[1, 1].set_title('Closing Image')
axes[1, 1].axis('off')

# 隐藏多余的子图
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()
