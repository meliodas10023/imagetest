import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取图像并预处理（灰度化 + 去噪）
image = cv2.imread('images/ts.jpg')  # 替换为你的图像路径

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)




