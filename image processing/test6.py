# 数字图像处理期末课程设计1：基于车道信息的违法车辆车牌识别
import cv2
import numpy as np
import matplotlib.pyplot as plt


# 展示图片的函数
def pic_display(dis_name, dis_image):
    cv2.imshow(dis_name, dis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 1.车牌定位
def license_region(image):
    img_b = cv2.split(image)[0]
    img_g = cv2.split(image)[1]
    img_r = cv2.split(image)[2]

    # 彩色信息特征初步定位：车牌定位并给resize后的图像二值化赋值
    standard_b = 138
    standard_g = 63
    standard_r = 23
    standard_threshold = 50
    img_test = image.copy()
    for i in range(img_test.shape[0]):
        for j in range(img_test.shape[1]):
            # 提取与给定的r、g、b阈值相差不大的点(赋值为全白)
            if abs(img_b[i, j] - standard_b) < standard_threshold \
                    and abs(img_g[i, j] - standard_g) < standard_threshold \
                    and abs(img_r[i, j] - standard_r) < standard_threshold:
                img_test[i, j, :] = 255
            # 其他所有的点赋值为全黑
            else:
                img_test[i, j, :] = 0
    pic_display('img_binary', img_test)

    # 基于数学形态学进一步精细定位车牌区域
    kernel = np.ones((3, 3), np.uint8)
    img_resize_dilate = cv2.dilate(img_test, kernel, iterations=5)  # 膨胀操作
    img_resize_erosion = cv2.erode(img_resize_dilate, kernel, iterations=5)  # 腐蚀操作
    pic_display('img_resize_erosion', img_resize_erosion)

    # cv2.cvtColor()函数将 三通道 的二值化图像转变为 单通道 的二值化图像
    img1 = cv2.cvtColor(img_resize_erosion, cv2.COLOR_BGR2GRAY)
    contours = cv2.findContours(img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]

    # 通过区域面积，宽高比例的方式进一步筛选车牌区域
    MIN_AREA = 200  # 设定矩形的最小区域，用于去除无用的噪声点
    car_contours = []
    for cnt in contours:  # contours是长度为18的一个tuple(元组)
        # 框选 生成最小外接矩形 返回值（中心(x,y), (宽,高), 旋转角度）rect[0]：矩形中心点坐标；rect[1]：矩形的高和宽；rect[2]：矩形的旋转角度
        rect = cv2.minAreaRect(cnt)
        area_width, area_height = rect[1]
        # 计算最小矩形的面积，初步筛选
        area = rect[1][0] * rect[1][1]  # 最小矩形面积
        if area > MIN_AREA:
            if area_width < area_height:  # 选择宽小于高的区域进行宽和高的置换
                area_width, area_height = area_height, area_width
            # 求出宽高之比(要求矩形区域长宽比在2到5.5之间，其他的排除)
            wh_ratio = area_width / area_height
            if 2 < wh_ratio < 5.5:
                car_contours.append(rect)
                box = cv2.boxPoints(rect)  # 存放最小矩形的四个顶点坐标(先列后行的顺序)
                box = np.int0(box)  # 去除小数点，只保留整数部分
    region_out = box
    return region_out


# 读取需检测的图片
img = cv2.imread('images/ps1.png')


# 原始图片过大，压缩为原始图像尺寸的1/3，方便opencv-python出图展示
# img_resize = cv2.resize(img, None, fx=1 / 3, fy=1 / 3, interpolation=cv2.INTER_CUBIC)

# 对原始图片进行平滑和滤波处理
# 高斯平滑
img_resize_gaussian = cv2.GaussianBlur(img, (5, 5), 1)
# 中值滤波
img_resize_median = cv2.medianBlur(img_resize_gaussian, 3)

# 定位车牌区域并绘图展示
region = license_region(img_resize_median)
# 在原始图像中用红色方框标注
img_showRect = img.copy()
img_showRect = cv2.drawContours(img_showRect, [region], 0, (0, 0, 255), 2)
pic_display('img_showRect', img_showRect)

# 将车牌区域提取出来
region_real = region * 3
car_region = img[np.min(region_real[:, 1]):np.max(region_real[:, 1]) + 5,
             np.min(region_real[:, 0]):np.max(region_real[:, 0]) + 10, :]
pic_display('car_region', car_region)
