import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('images/1.jpg')
cv2.imshow('original image', img)
cv2.waitKey(0)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray image', gray_img)
cv2.waitKey(0)



# 1.log 检测器
blur=cv2.GaussianBlur(gray_img,(3,3),1,1)#核尺寸通过对图像的调节自行定义
#调用Laplacian 算法的OpenCV库函数进行图像轮廓提取
result = cv2.Laplacian(blur,cv2.CV_16S,ksize=1)
LOG = cv2.convertScaleAbs(result)#得到 LOG 算法处理结果
cv2.imshow('LOG',LOG)
cv2.waitKey(0)
plt.figure(figsize=(12, 6))
plt.subplot(141), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.subplot(142), plt.imshow(gray_img, cmap='gray'), plt.title('gray_img')
plt.subplot(143), plt.imshow(blur, cmap='gray'), plt.title('blur')
plt.subplot(144), plt.imshow(LOG, cmap='gray'), plt.title('LOG')
plt.tight_layout()
plt.show()

# 读取图像并转换为灰度图
image = cv2.imread('images/1.jpg', cv2.IMREAD_GRAYSCALE)

# 使用 Scharr 算子计算水平和垂直方向的梯度
grad_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
grad_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)

# 计算梯度幅值
grad_magnitude = cv2.magnitude(grad_x.astype(np.float32), grad_y.astype(np.float32))

# 显示结果
plt.figure(figsize=(12, 6))
plt.subplot(141), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(142), plt.imshow(grad_x, cmap='gray'), plt.title('Scharr X')
plt.subplot(143), plt.imshow(grad_y, cmap='gray'), plt.title('Scharr Y')
plt.subplot(144), plt.imshow(grad_magnitude, cmap='gray'), plt.title('Gradient Magnitude')
plt.tight_layout()
plt.show()

# 3. Canny边缘检测器
#图像高斯滤波去噪
blur=cv2.GaussianBlur(gray_img,(7,7),1,1)
#Canny 算子进行边缘提取
Canny = cv2.Canny(blur, 50, 150)
plt.figure(figsize=(12, 6))
plt.subplot(141), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.subplot(142), plt.imshow(gray_img, cmap='gray'), plt.title('gray_img')
plt.subplot(143), plt.imshow(blur, cmap='gray'), plt.title('blur')
plt.subplot(144), plt.imshow(Canny, cmap='gray'), plt.title('Canny')
plt.tight_layout()
plt.show()
