import cv2
import numpy as np


def detect_stem(mask):
    """检测茎区域：在主区域顶部搜索小连通区域"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)
    roi = mask[max(y - 20, 0):y, x:x + w]  # 主区域顶部上方20像素内搜索茎
    return cv2.countNonZero(roi) > 20  # 茎区域像素数阈值


def convexity_defects_count(contour):
    """计算凸包缺陷数量（凹凸不平程度）"""
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull.shape[0] < 3:
        return 0
    defects = cv2.convexityDefects(contour, hull)
    return len(defects) if defects is not None else 0


# 读取图像并检查
img = cv2.imread('images/6水果识别图像.jpg')
if img is None:
    print("错误：无法读取图像！")
    exit()

# 1. 预处理
blur = cv2.GaussianBlur(img, (5, 5), 0)
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV_FULL)  # 使用0-255范围的HSV

# 2. 二值化（基于饱和度+亮度）
_, s_thresh = cv2.threshold(hsv[:, :, 1], 50, 255, cv2.THRESH_BINARY)
_, v_thresh = cv2.threshold(hsv[:, :, 2], 100, 255, cv2.THRESH_BINARY)
binary = cv2.bitwise_and(s_thresh, v_thresh)

# 3. 形态学处理
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)

# 4. 连通区域分析
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, 8, cv2.CV_32S)
print(f"检测到连通区域数量: {num_labels - 1}")

# 5. 特征提取与分类
output = img.copy()
classified = set()  # 记录已分类的标签

for i in range(1, num_labels):
    mask = (labels == i).astype(np.uint8)
    area = stats[i, cv2.CC_STAT_AREA]

    # 颜色特征
    mean_hue = cv2.mean(hsv[:, :, 0], mask)[0]  # 色调 (0-255)
    mean_sat = cv2.mean(hsv[:, :, 1], mask)[0]  # 饱和度
    mean_val = cv2.mean(hsv[:, :, 2], mask)[0]  # 亮度

    # 形状特征
    contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not contour:
        continue
    contour = contour[0]
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

    # 伸长度
    rect = cv2.minAreaRect(contour)
    w, h = rect[1]
    elongation = max(w, h) / min(w, h) if min(w, h) > 0 else 0

    # 凹凸特征
    defects_num = convexity_defects_count(contour)

    # 茎检测
    has_stem = detect_stem(mask)

    # 分类逻辑
    fruit = None

    # 红色系水果
    if 0 <= mean_hue <= 15 or 240 <= mean_hue <= 255:
        if mean_sat > 150 and circularity > 0.7:  # 高饱和度+圆形
            fruit = "Apple"
        elif 80 < mean_sat <= 150:
            if defects_num > 5:  # 凹凸不平
                fruit = "Litchi" if mean_val > 160 else "Strawberry"
            else:
                fruit = "Peach" if area < 5000 else None

    # 黄色系水果
    elif 20 <= mean_hue <= 40:
        if elongation > 4:
            fruit = "Banana"
        elif elongation > 2:
            fruit = "Mango"
        elif has_stem:  # 梨的茎特征
            fruit = "Pear"

    # 如果分类成功，标记区域
    if fruit:
        classified.add(i)
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[
            i, cv2.CC_STAT_HEIGHT]
        cv2.putText(output, fruit, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# 6. 剩余未分类区域标记为菠萝
for i in range(1, num_labels):
    if i not in classified:
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[
            i, cv2.CC_STAT_HEIGHT]
        cv2.putText(output, "Pineapple", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# 保存结果
cv2.imwrite('result.jpg', output)

# 显示结果（调试用）
cv2.imshow('Result', output)
cv2.waitKey(0)
cv2.destroyAllWindows()