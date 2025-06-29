import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取灰度图像
img = cv2.imread('images/1.jpg', 0)

# 计算原始图像的灰度直方图
hist = cv2.calcHist([img], [0], None, [256], [0, 255])

# 直方图均衡化
equ = cv2.equalizeHist(img)

# 计算均衡化后的灰度直方图
equ_hist = cv2.calcHist([equ], [0], None, [256], [0, 255])

# 创建一个2x2的子图布局
plt.figure(figsize=(10, 8))

# 显示原始图像
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')
#
# # 显示原始图像的直方图
# plt.subplot(2, 2, 2)
# plt.plot(hist, color='blue')
# plt.title('Original Histogram')
# plt.xlim([0, 255])
#
# # 显示均衡化后的图像
# plt.subplot(2, 2, 3)
# plt.imshow(equ, cmap='gray')
# plt.title('Equalized Image')
# plt.axis('off')
#
# # 显示均衡化后的直方图
# plt.subplot(2, 2, 4)
# plt.plot(equ_hist, color='blue')
# plt.title('Equalized Histogram')
# plt.xlim([0, 255])
#
# # 调整子图之间的间距
# plt.tight_layout()
#
# # 显示所有图像
# plt.show()



# # cv2.imwrite('equ1',equ)
# # 空间域滤波
# # 一.平滑滤波
# # 1.均值滤波
# mean_blur = cv2.blur(img, (3, 5))  # 模板大小为 3*5, 模板的大小是可以设定的
#
# # box = cv2.boxFilter(img, -1, (3, 5))
# cv2.imshow('mean_blur', mean_blur)
# cv2.waitKey(0)
# #
# # 2.高斯模糊滤波
# gaussian_blur = cv2.GaussianBlur(equ, (5, 5), 0)
# cv2.imshow('gaussian_blur', gaussian_blur)
# cv2.waitKey(0)
# # 3.中值滤波
# median_blur = cv2.medianBlur(equ, 5)
# cv2.imshow('median_blur', median_blur)
# cv2.waitKey(0)
# # '''

# 二.锐化滤波、
# 1. Roberts算子

img_build = cv2.imread('images/1.jpg',0)
cv2.imshow('img',img_build)

#图像高斯滤波去噪
blur=cv2.GaussianBlur(img_build,(3,3),1,1) #核尺寸通过对图像的调节自行定义
#图像阈值化处理
ret,thresh1=cv2.threshold(blur,127,255,cv2.THRESH_BINARY)  #二进制阈值化
#调用Roberts算法的OpenCV库函数进行图像轮廓提取
# kernelx_Roberts = np.array([[-1,0],[0,1]], dtype=int)
# kernely_Roberts = np.array([[0,-1],[1,0]], dtype=int)
# x = cv2.filter2D(thresh1, cv2.CV_16S, kernelx_Roberts)
# y = cv2.filter2D(thresh1, cv2.CV_16S, kernely_Roberts)
# #转uint8
# absX = cv2.convertScaleAbs(x)
# absY = cv2.convertScaleAbs(y)
# Roberts = cv2.addWeighted(absX,0.5,absY,0.5,0)
# cv2.imshow("Roberts",Roberts)
# cv2.waitKey()
#
#
# # 2.prewitt算子
# kernelx_prewitt = np.array([[1,1,1],[0,0,0],[-1,-1,-1]],dtype=int)
# kernely_prewitt = np.array([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=int)
# x = cv2.filter2D(thresh1, -1, kernelx_prewitt)
# y = cv2.filter2D(thresh1, -1, kernely_prewitt)
# #转uint8
# absX = cv2.convertScaleAbs(x)
# absY = cv2.convertScaleAbs(y)
# prewitt = cv2.addWeighted(absX,0.5,absY,0.5,0)
# cv2.imshow("prewitt",prewitt)
# cv2.waitKey()

# 3.Sobel算子
# x_Sobel = cv2.Sobel(thresh1, cv2.CV_16S, 1, 0) #对x求一阶导
# y_Sobel = cv2.Sobel(thresh1, cv2.CV_16S, 0, 1) #对y求一阶导
# absX = cv2.convertScaleAbs(x_Sobel)  #对x取绝对值，并将图像转换为8位图
# absY = cv2.convertScaleAbs(y_Sobel)   #对y取绝对值，并将图像转换为8位图
# Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
# cv2.imshow("Sobel",Sobel)
# cv2.waitKey()
#


# 三.频域滤波
 #1.低频滤波
 # ①理想低通滤波

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转换为灰度图
image = cv2.imread('images/ts.jpg', cv2.IMREAD_GRAYSCALE)

# 将图像转换为频域
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)

# 创建理想低通滤波器
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2
mask = np.ones((rows, cols), np.uint8)
# D0 = 30  # 截止频率
# for i in range(rows):
#     for j in range(cols):
#         d = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
#         if d <= D0:
#             mask[i, j] = 0
#
# # 应用滤波器
# fshift = fshift * mask
#
# # 将图像转换回空间域
# ishift = np.fft.ifftshift(fshift)
# img_back = np.fft.ifft2(ishift)
# img_back = np.abs(img_back)
# #
# # 显示结果
# plt.figure(figsize=(12, 6))
# plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image')
# plt.subplot(122), plt.imshow(img_back, cmap='gray'), plt.title('Ideal Highpass Filtered Image')
# plt.tight_layout()
# plt.show()





# # ② 巴特沃斯低通滤波
# 创建巴特沃斯低通滤波器
# n = 2  # 阶数
# D0 = 30  # 截止频率
# mask = np.zeros((rows, cols), np.float32)
# for i in range(rows):
#     for j in range(cols):
#         d = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
#         mask[i, j] = 1 / (1 + (D0 / d) ** (2 * n))
#
# # 应用滤波器
# fshift = fshift * mask
#
# # 将图像转换回空间域
# ishift = np.fft.ifftshift(fshift)
# img_back = np.fft.ifft2(ishift)
# img_back = np.abs(img_back)
#
# # 显示结果
# plt.figure(figsize=(12, 6))
# plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image')
# plt.subplot(122), plt.imshow(img_back, cmap='gray'), plt.title('Butterworth Highpass Filtered Image')
# plt.tight_layout()
# plt.show()


# #③高斯低通滤波

# # 创建高斯低通滤波器
D0 = 30  # 截止频率
mask = np.ones((rows, cols), np.float32)
for i in range(rows):
    for j in range(cols):
        d = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
        mask[i, j] = 1-np.exp(-d**2 / (2 * D0**2))

# 应用滤波器
fshift = fshift * mask

# 将图像转换回空间域
ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(ishift)
img_back = np.abs(img_back)

# 显示结果
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(img_back, cmap='gray'), plt.title('Gaussian Highpass Filtered Image')
plt.tight_layout()
plt.show()








