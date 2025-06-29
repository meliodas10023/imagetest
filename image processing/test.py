import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取图像并预处理（灰度化 + 去噪）
image = cv2.imread('images/ts.jpg')  # 替换为你的图像路径

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

cv2.imshow("blurred", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()




