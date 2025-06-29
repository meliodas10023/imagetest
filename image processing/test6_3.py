import cv2
import numpy as np
import matplotlib.pyplot as plt
import os



# 1. 图像预处理
# def preprocess_image(image_path):
#     # 读取图像并转为RGB格式
#     img = cv2.imread(image_path)
#
#     # 灰度化
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     # 直方图均衡化
#     equalized = cv2.equalizeHist(gray)
#     # 中值滤波去噪
#     filtered = cv2.medianBlur(equalized, 3)
#
#     return img, equalized

#手动调整HSV阈值
def adjust_hsv_threshold(image):
    def nothing(x):
        pass

    cv2.namedWindow("HSV Adjust")
    cv2.createTrackbar("H Min", "HSV Adjust", 190, 255, nothing)
    cv2.createTrackbar("H Max", "HSV Adjust", 245, 255, nothing)
    cv2.createTrackbar("S Min", "HSV Adjust", int(0.35 * 255), 255, nothing)
    cv2.createTrackbar("S Max", "HSV Adjust", 255, 255, nothing)
    cv2.createTrackbar("V Min", "HSV Adjust", int(0.3 * 255), 255, nothing)
    cv2.createTrackbar("V Max", "HSV Adjust", 255, 255, nothing)

    while True:
        h_min = cv2.getTrackbarPos("H Min", "HSV Adjust")
        h_max = cv2.getTrackbarPos("H Max", "HSV Adjust")
        s_min = cv2.getTrackbarPos("S Min", "HSV Adjust")
        s_max = cv2.getTrackbarPos("S Max", "HSV Adjust")
        v_min = cv2.getTrackbarPos("V Min", "HSV Adjust")
        v_max = cv2.getTrackbarPos("V Max", "HSV Adjust")

        lower_hsv = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper_hsv = np.array([h_max, s_max, v_max], dtype=np.uint8)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# 2. 车牌区域检测（基于颜色和形态学）
def locate_license_plate(img):
    # 转换到HSV颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # 定义蓝色车牌范围（根据实际调整）
    lower_hsv = np.array([80, 94, 116], dtype=np.uint8)
    upper_hsv = np.array([127, 206, 178], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # cv2.imshow("HSV Mask", mask)
    # cv2.waitKey(0)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 筛选符合尺寸比例的轮廓（车牌近似矩形）
    candidate_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h  # 车牌宽高比一般在2~5之间
        area = cv2.contourArea(cnt)

        # 设置过滤条件：面积和宽高比
        if area > 500 and 2 < aspect_ratio < 5:
            candidate_contours.append((x, y, w, h))

    # 如果没有找到符合条件的轮廓，返回None
    if not candidate_contours:
        return None

    # 按面积排序，取最大一个
    candidate_contours.sort(key=lambda x: x[2] * x[3], reverse=True)
    x, y, w, h = candidate_contours[0]

    # 返回裁剪出的车牌区域
    return img[y:y + h, x:x + w]


# 3. 车牌预处理
def binarize_plate(plate_img):
    # 自适应阈值二值化
    binary = cv2.threshold(plate_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # # 腐蚀去除小噪点
    # kernel = np.ones((2, 2), np.uint8)
    # eroded = cv2.erode(binary, kernel, iterations=1)
    return binary




def segment_characters(binary_image):
    # 计算纵轴投影
    projection = np.sum(binary_image, axis=0)

    # 分割字符
    characters = []
    start = 0
    for i in range(1, len(projection)):
        if projection[i] == 0 and projection[i - 1] != 0:
            char_img = binary_image[:, start:i]

            # 调整尺寸为 25x15
            resized_char = cv2.resize(char_img, (15, 25), interpolation=cv2.INTER_AREA)

            # resized_char = cv2.bitwise_not(resized_char)

            characters.append(resized_char)

            start = i

    # 删除第一个和最后一个字符
    if len(characters) > 2:
        characters = characters[1:-1]
    else:
        characters = []  # 如果不足3个，就清空（根据需求可调整）

        # 对每个字符应用 template_segmentation
    processed_characters = []
    for char in characters:
        # 应用模板分割函数
        processed_char = template_segmentation(char)
        processed_characters.append(processed_char)

    # 显示分割结果（可选）
    for idx, char in enumerate(processed_characters):
        cv2.namedWindow(f"Processed Character {idx + 1}", cv2.WINDOW_NORMAL)
        cv2.imshow(f"Processed Character {idx + 1}", char)
        cv2.resizeWindow(f"Processed Character {idx + 1}", 100, 200)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return processed_characters

# def preprocess_template(template_img):
#     """
#     对模板图像进行预处理，使其与分割后的字符图像相似
#     :param template_img: 输入的模板图像 (灰度图)
#     :return: 预处理后的模板图像 (15x25, 黑底白字)
#     """
#     # 步骤1：缩放至统一尺寸
#     resized = cv2.resize(template_img, (15, 25), interpolation=cv2.INTER_AREA)
#
#     # 步骤2：反转颜色 → 黑底白字
#     inverted = cv2.bitwise_not(resized)
#
#     # 步骤3：二值化处理（增强对比度）
#     _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#     # 步骤4（可选）：形态学操作去除噪点
#     kernel = np.ones((2, 2), np.uint8)
#     cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
#
#     return cleaned




def template_segmentation(origin_img):
    # 模板分割函数，只针对单个字符，用于去除其周围的边缘，并resize
    # 提取字符各列满足条件(有两个255的单元格)的索引
    col_index = []
    for col in range(origin_img.shape[1]):  # 对于图像的所有列
        if np.sum(origin_img[:, col]) >= 2*255:
            col_index.append(col)
    col_index = np.array(col_index)
    # 提取字符各行满足条件(有两个255的单元格)的索引
    row_index = []
    for row in range(origin_img.shape[0]):
        if np.sum(origin_img[row, :]) >= 2*255:
            row_index.append(row)
    row_index = np.array(row_index)
    # 按索引提取字符(符合条件的行列中取min-max)，并resize到25*15大小
    output_img = origin_img[np.min(row_index):np.max(row_index)+1, np.min(col_index):np.max(col_index)+1]
    output_img = np.uint8(output_img)
    if col_index.shape[0] <= 3 or row_index.shape[0] <= 3:
        output_img = origin_img[np.min(row_index):np.max(row_index)+1, np.min(col_index):np.max(col_index)+1]
        pad_row1 = np.int8(np.floor((25 - output_img.shape[0]) / 2))
        pad_row2 = np.int8(np.ceil((25 - output_img.shape[0]) / 2))
        pad_col1 = np.int8(np.floor((15 - output_img.shape[1]) / 2))
        pad_col2 = np.int8(np.ceil((15 - output_img.shape[1]) / 2))
        output_img = np.pad(output_img, ((pad_row1, pad_row2), (pad_col1, pad_col2)), 'constant',
                            constant_values=(0, 0))
        output_img = np.uint8(output_img)
    else:
        output_img = cv2.resize(output_img, (15, 25), interpolation=0)
    return output_img

def template_array_generator(template_path, template_size):
    template_img_out = np.zeros([template_size, 25, 15], dtype=np.uint8)
    index = 0
    files = os.listdir(template_path)
    for file in files:
        template_img = cv2.imdecode(np.fromfile(template_path + '/' + file, dtype=np.uint8), -1)
        template_img_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        template_img_binary = cv2.threshold(template_img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        template_img_binary = 255-template_img_binary   # 模板给出的与车牌上的是相反的,所有用255相减进行匹配
        template_img_out[index, :, :] = template_segmentation(template_img_binary)
        index = index + 1
    return template_img_out


def recognize_character(char_img, templates, labels):

    min_diff = float('inf')
    matched_char = ''

    for i, template in enumerate(templates):
        # 计算绝对差值总和
        diff = np.sum(np.abs(char_img.astype("int32") - template.astype("int32")))

        if diff < min_diff:
            min_diff = diff
            matched_char = labels[i]

    return matched_char


def read_labels(label_file, encoding='gbk'):
    with open(label_file, 'r', encoding=encoding) as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    return labels

# 主流程
if __name__ == "__main__":
    # 1. 预处理

    image = cv2.imread('images/ps1.png')
    image = cv2.resize(image, (800, 600))

    cv2.imshow('Original Image', image)
    cv2.waitKey(0)

    plate_region = locate_license_plate(image)
    cv2.imshow('Detected License Plate', plate_region)
    cv2.waitKey(0)

    binary_plate = binarize_plate(plate_region)
    cv2.imshow('Binarized Plate', binary_plate)
    cv2.waitKey(0)


    characters = segment_characters(binary_plate)
    print("Number of characters:", len(characters))
    print(type(characters))

    # # 使用matplotlib绘制7个子图
    # fig, axs = plt.subplots(1, 7, figsize=(14, 2))  # 1行7列的子图布局
    #
    # for i, char in enumerate(characters):  # 只取前7个字符
    #     axs[i].imshow(char, cmap='gray')  # 灰度显示
    #     axs[i].axis('off')  # 关闭坐标轴
    #
    # plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    # plt.show()

    # 进行字符识别
    # 读取字符标签
    chinese_labels = read_labels('images/5-carNumber/character.txt')
    letter_labels = read_labels('images/5-carNumber/letter.txt')
    number_labels = read_labels('images/5-carNumber/num.txt')

    # 加载模板（假设每个类别有固定数量的模板）
    chinese_templates = template_array_generator('images/5-carNumber/character', len(chinese_labels))
    letter_templates = template_array_generator('images/5-carNumber/letter', len(letter_labels))
    number_templates = template_array_generator('images/5-carNumber/num', len(number_labels))

    # 显示第一个模板
    first_template = chinese_templates[0]
    cv2.imshow("First Template", first_template)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    recognized_chars = []


# 示例运行
