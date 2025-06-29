import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_glcm_features(gray_img, mask, distance=1, angle=0):
    # 灰度级为8位，方向默认为 0，即水平方向
    levels = 256
    glcm = np.zeros((levels, levels), dtype=np.float64)
    dx = int(np.round(np.cos(angle * np.pi / 180))) * distance
    dy = int(np.round(-np.sin(angle * np.pi / 180))) * distance

    rows, cols = gray_img.shape
    for y in range(rows - abs(dy)):
        for x in range(cols - abs(dx)):
            if mask[y, x] == 0:
                continue
            y2, x2 = y + dy, x + dx
            if 0 <= x2 < cols and 0 <= y2 < rows and mask[y2, x2] == 1:
                i = gray_img[y, x]
                j = gray_img[y2, x2]
                glcm[i, j] += 1
    # 归一化
    total = np.sum(glcm)
    if total == 0:
        return 0, 0, 0  # 无特征
    glcm /= total
    # 特征计算
    contrast = 0
    energy = 0
    homogeneity = 0
    for i in range(levels):
        for j in range(levels):
            p = glcm[i, j]
            contrast += (i - j) ** 2 * p
            energy += p ** 2
            homogeneity += p / (1 + abs(i - j))

    return contrast, energy, homogeneity


def classify_fruit(mean_h, mean_s, mean_v, area, circularity, elongation, contrast, energy, homogeneity):
    # 1. 优先检测菠萝（绿色+大面积）
    if 40 < mean_h < 60 and area > 20000:
        return 'Pineapple'

    # 2. 红色水果（H在0-20或160-180）
    if 40 <= mean_h <=90 or 0 <= mean_h <= 30:

        if contrast >200  and  0.4>homogeneity  >0.3 and circularity > 0.85:
            return 'Apple'
        elif energy <0.003 and contrast <200.00:
            return 'Peach'
        elif mean_v > 200 and area < 6000 and contrast >700 :
            return 'Strawberry'
        elif 500<contrast <700 and mean_s > 100:
            return 'Lychee'

    # 3. 黄色水果（H在15-40）
    if 20 <= mean_h <= 30:
        if circularity < 0.50 and elongation > 2 and energy >0.003:
            return 'Banana'
        elif  elongation > 2 and  circularity >0.5:
            return 'Mango'
        elif contrast >290 and mean_s < 150:
            return 'Pear'


def image_display(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(0)


#  读取图像并预处理（灰度化 + 高斯去噪）
image = cv2.imread('images/6水果识别图像.jpg')  # 替换为你的图像路径
image = cv2.resize(image, (640, 480))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Canny边缘检测
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
dilated = cv2.dilate(closed, kernel, iterations=2)

#找轮廓并填充闭区域（获取完整闭合区域）
contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filled = np.zeros_like(gray)
cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

# 显示各阶段结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 5, 1), plt.title("Original Image"), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.axis('off')  # 关键修改：BGR转RGB
plt.subplot(1, 5, 2), plt.title("Original Gray"), plt.imshow(gray, cmap='gray'), plt.axis('off')
plt.subplot(1, 5, 3), plt.title("Edges"), plt.imshow(edges, cmap='gray'), plt.axis('off')
plt.subplot(1, 5, 4), plt.title("Morph Closed"), plt.imshow(dilated, cmap='gray'), plt.axis('off')
plt.subplot(1, 5, 5), plt.title("Filled Contours"), plt.imshow(filled, cmap='gray'), plt.axis('off')
plt.tight_layout()
plt.show()


# 连通区域分析
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filled, 8, cv2.CV_32S)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
for i in range(1, num_labels):
    mask = (labels == i).astype(np.uint8)
    area = stats[i, cv2.CC_STAT_AREA]
    if area < 100:
        continue

    # 颜色特征
    mean_h = np.mean(hsv[:, :, 0][mask == 1])
    mean_s = np.mean(hsv[:, :, 1][mask == 1])
    mean_v = np.mean(hsv[:, :, 2][mask == 1])

    # 形状特征
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea) if contours else None
    if not cnt.all():
        continue

    perimeter = cv2.arcLength(cnt, True)
    area_cnt = cv2.contourArea(cnt)
    circularity = (4 * np.pi * area_cnt) / (perimeter ** 2) if perimeter > 0 else 0

    # 伸长度
    rect = cv2.minAreaRect(cnt)
    major_length = max(rect[1])
    minor_length = min(rect[1])
    elongation = major_length / minor_length if minor_length > 0 else 0

    #纹理特征
    contrast, energy, homogeneity = compute_glcm_features(gray, mask)
    # 分类并打印特征
    fruit_type = classify_fruit(mean_h, mean_s, mean_v, area_cnt, circularity, elongation, contrast, energy, homogeneity)
    print(
        f"Region {i}: {fruit_type}, H={mean_h:.1f}, S={mean_s:.1f}, V={mean_v:.1f}, C={circularity:.2f}, E={elongation:.2f},A ={area:.1f}，"
        f" Contrast={contrast:.2f}, Energy={energy:.3f}, Homogeneity={homogeneity:.3f}")

    # 标注结果
    # 标注位置中心
    x, y = int(centroids[i][0]), int(centroids[i][1])

    # 绘制水果名称
    if fruit_type is not None:
        cv2.putText(image, fruit_type, (x - 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # 获取边界框并画出红色矩形框
        x_rect, y_rect, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x_rect, y_rect), (x_rect + w, y_rect + h), (0, 0, 255), 2)

cv2.imshow('Result', image)
cv2.waitKey(0)