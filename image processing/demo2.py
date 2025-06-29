import cv2
import numpy as np

# 读取原始车辆图像
img = cv2.imread('images/ps1.png')

# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 高斯模糊去噪
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Canny 边缘检测
edges = cv2.Canny(blurred, 50, 150)

# 霍夫变换检测直线
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

# 根据检测到的直线确定车牌角点
if lines is not None:
    # 对直线进行分析，确定车牌角点
    # 这里需要根据实际情况进行复杂的分析和计算
    # 提取角点坐标作为源点集
    # ...
    # 假设得到的角点为 pts
    # 构建目标点集
    width = 300
    height = 100
    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    # 计算透视变换矩阵并进行变换
    M = cv2.getPerspectiveTransform(pts, dst_pts)
    corrected_img = cv2.warpPerspective(img, M, (width, height))
    # 显示摆正后的图像
    cv2.imshow('Corrected Vehicle', corrected_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()