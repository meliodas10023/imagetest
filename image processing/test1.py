'''
import cv2 as cv
img = cv.imread('1.jpg')    
print(img.shape)    
#引入OpenCV库 
#使用imread 函数读取图像，并以numpy数组形式储存 
#查看图像的大小。返回的元组（touple）中的三个数依次表示高度、宽度和通道数
print(img.dtype)    
#查看图片的类型 
cv.imshow('img',img)    
#使用imshow函数显示图像，第一个参数是窗口名称（可不写），第二个参数是要显示的图像的名称，一定要写
cv.waitKey(0)
#可以让窗口一直显示图像直到按下任意按键 
img_GRAY = cv.cvtColor(img,cv.COLOR_BGR2GRAY)   #使用 cv.cvtColor 函数转换色彩空间，参数‘cv.COLOR_BGR2GRAY’表示从RGB空间转换到灰度空间
cv.imshow('gray',img_GRAY)
cv.imwrite("images/gray1.jpg", img_GRAY)
cv.waitKey(0)
ret,thresh = cv.threshold(img_GRAY,127,255,cv.THRESH_BINARY)    
#使用cv.threshold函数进行图像阈值处理，参数‘cv.THRESH_BINARY’代表了阈值的类型，127为阈值
cv.imshow('threshold',thresh) 
cv.waitKey(0)
res = cv.resize(img,None,fx=2,fy=2,interpolation=cv.INTER_CUBIC)
cv.imshow('resize',res) 
cv.waitKey(0)
cv.imwrite('result.jpg',res)'''


import cv2 as cv
import matplotlib.pyplot as plt

# 引入OpenCV库
# 使用imread 函数读取图像，并以numpy数组形式储存
img = cv.imread('images/1.jpg')
# print(img.shape)  # 查看图像的大小。返回的元组（tuple）中的三个数依次表示高度、宽度和通道数
# print(img.dtype)  # 查看图片的类型

# 转换为灰度图像
img_GRAY = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 阈值处理
ret, thresh = cv.threshold(img_GRAY, 127, 255, cv.THRESH_BINARY)

# 调整图像大小
res = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)

# 保存灰度图像
cv.imwrite("images/gray2.jpg", img_GRAY)

# 保存调整大小后的图像
cv.imwrite('result.jpg', res)

# 平移操作
rows, cols = img.shape[:2]
M = cv.getRotationMatrix2D((cols / 2, rows / 2), 0, 1)  # 创建平移矩阵
M[0, 2] += 50  # 水平平移50像素
M[1, 2] += 30  # 垂直平移30像素
translated = cv.warpAffine(img, M, (cols, rows))

# 翻转操作
flipped = cv.flip(img, 1)  # 1表示水平翻转，0表示垂直翻转，-1表示同时水平和垂直翻转

# 创建一个2x2的子图布局
fig, axs = plt.subplots(2, 3, figsize=(10, 10))

# 显示原始图像
axs[0, 0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
axs[0, 0].axis('off')  # 关闭坐标轴

# 显示灰度图像
axs[0, 1].imshow(img_GRAY, cmap='gray')
axs[0, 1].set_title('Gray Image')
axs[0, 1].axis('off')  # 关闭坐标轴

# 显示阈值图像
axs[0, 2].imshow(thresh, cmap='gray')
axs[0, 2].set_title('Threshold Image')
axs[0, 2].axis('off')

# 显示调整大小后的图像
axs[1, 0].imshow(cv.cvtColor(res, cv.COLOR_BGR2RGB))
axs[1, 0].set_title('Resized Image')
axs[1, 0].axis('off')

axs[1, 1].imshow(cv.cvtColor(translated, cv.COLOR_BGR2RGB))
axs[1, 1].set_title('Translated Image')
axs[1, 1].axis('off')

axs[1, 2].imshow(cv.cvtColor(flipped, cv.COLOR_BGR2RGB))
axs[1, 2].set_title('Flipped Image')
axs[1, 2].axis('off')


# 调整子图之间的间距
plt.tight_layout()

# 显示所有图像
plt.show()




