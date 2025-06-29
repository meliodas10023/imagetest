import math
import cv2
import numpy as np


# 数据预处理函数
def preProcess(Img):
    # 调整图片长宽像素比调整显示大小
    # 列为600行为400
    OriResize = cv2.resize(Img, (600, 400))
    # 转HSV颜色空间利用饱和度去除白色背景
    HSV = cv2.cvtColor(OriResize, cv2.COLOR_BGR2HSV)
    # 通过实验发现所使用背景可利用s_min=35的饱和度去除
    lower = np.array([0, 35, 0])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(HSV, lower, upper)
    # 进行高斯模糊
    blur = cv2.GaussianBlur(mask, (7, 7), 1)
    # 利用canny算子提取边缘
    canny = cv2.Canny(blur, 50, 50)
    # 进行一次闭运算,去除杂点
    kenel = np.ones((3, 3))
    mask1 = cv2.dilate(canny, kenel, iterations=1)
    mask2 = cv2.erode(mask1, kenel, iterations=1)
    return OriResize, mask2, blur


# 读入群体识别目标图

OriImg = cv2.imread('images/6水果识别图像.jpg')
# 数据预处理
Ryuanshi, bianyuan, Obinary = preProcess(OriImg)
# 为RGB提取做原始数据备份
RYcopy = np.copy(Ryuanshi)
# 获取多目标轮廓信息
contours, hierarchy = cv2.findContours(bianyuan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# 利用约束规则与自设近似模型评价识别
for cnt in contours:
    # 利用封闭面积去除小区域边界影响，获取精确目标边界
    area = cv2.contourArea(cnt)
    if area < 100:
        continue
    else:
        # 构建特征向量
        feature = []
        # 未识别出的目标标识 unclassic
        objectType = "unclassic"
        # 判别模型
        # 第一步：获取面积与周长的比值
        pri = cv2.arcLength(cnt, True)
        div = area / pri
        feature.append(div)
        # 第二步：获取R，G，B信息
        # 将边界的点集利用DP算法压缩且保证误差在0.02倍周长内，假定边界闭合,获取最小外接矩形的左上坐标与长宽
        approx = cv2.approxPolyDP(cnt, 0.02 * pri, True)
        x, y, k, h = cv2.boundingRect(approx)
        # 加入定位框比
        if k > h:
            feature.append(k / h)
        else:
            feature.append(h / k)
        # 查看坐标
        print(x, y)
        MultOJ = cv2.bitwise_and(RYcopy, RYcopy, mask=Obinary)
        Bg, Gg, Rg = cv2.split(RYcopy)
        B, G, R = cv2.split(MultOJ)
        # 取背景板值样本
        udB = 0
        udG = 0
        udR = 0
        for i in range(10):
            for j in range(10):
                udB += Bg[j, i]
                udG += Gg[j, i]
                udR += Rg[j, i]
        udB /= 100
        udG /= 100
        udR /= 100
        # 正式开始获取颜色特征
        fMB = 0
        fMG = 0
        fMR = 0
        Rcount = 0
        for i in range(x, x + k):
            for j in range(y, y + h):
                if udB - 8 < B[j, i] < udB + 8 and udG - 8 < G[j, i] < udG + 8 and udR - 8 < R[j, i] < udR + 8:
                    continue
                else:
                    fMB += B[j, i]
                    fMG += G[j, i]
                    fMR += R[j, i]
                    Rcount += 1
        fMB /= Rcount
        fMG /= Rcount
        fMR /= Rcount
        feature.append(fMB)
        feature.append(fMG)
        feature.append(fMR)
        # 求解方差
        fcB = 0
        fcG = 0
        fcR = 0
        for i in range(x, x + k):
            for j in range(y, y + h):
                if udB - 8 < B[j, i] < udB + 8 and udG - 8 < G[j, i] < udG + 8 and udR - 8 < R[j, i] < udR + 8:
                    continue
                else:
                    fcB += pow((B[j, i] - fMB), 2)
                    fcG += pow((G[j, i] - fMG), 2)
                    fcR += pow((R[j, i] - fMR), 2)
        fcB = math.sqrt(fcB / Rcount)
        fcG = math.sqrt(fcG / Rcount)
        fcR = math.sqrt(fcR / Rcount)
        feature.append(fcB)
        feature.append(fcG)
        feature.append(fcR)
        # 特征查看检验处
        print(feature)
        # 利用特征开始进行判别
        if feature[1] > 1.2:
            if abs(feature[2] - feature[3]) < 4:
                objectType = "grape"
            else:
                objectType = "banana"
        else:
            if abs(feature[2] - feature[3]) < 4:
                objectType = "apple"
            else:
                if feature[7] - feature[5] > 8 and feature[7] - feature[6] > 8:
                    objectType = "orange"
                else:
                    objectType = "kiwifruit"
        # 绘制最小外接矩形与其对应边界框
        cv2.drawContours(Ryuanshi, cnt, -1, (0, 0, 255), 1)
        cv2.rectangle(Ryuanshi, (x, y), (x + k, y + h), (0, 255, 0))
        # 将目标物的类别显示于目标物的最小外接矩形框的左上角附近
        cv2.putText(Ryuanshi, objectType, (x - 5, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
# 显示识别结果
cv2.imshow("Ryuanshi1", Ryuanshi)
cv2.waitKey(0)
