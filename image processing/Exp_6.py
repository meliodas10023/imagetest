import os
import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')

#定位车牌
def locate_license_plate(image):
    try:
        # 将图像转换为HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 定义HSV阈值，并确保为整数类型
        lower_hsv = np.array([80, 94, 116], dtype=np.uint8)
        upper_hsv = np.array([127, 206, 178], dtype=np.uint8)
        # 阈值处理
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        # 开运算去噪
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 找到车牌区域
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
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
            plate = image[y:y + h, x:x + w]
            # 显示定位的车牌
            cv2.imshow("定位的车牌", plate)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return plate
        else:
            print("未找到车牌区域")
            return None
    except Exception as e:
        print(f"处理图像时发生错误: {e}")
        return None

#灰度化二值化
def preprocess_license_plate(plate_image):
    # 灰度化
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    # 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 显示灰度化二值化后的车牌
    cv2.imshow("二值化后的车牌", binary)
    cv2.resize(binary, (800, 600))
    cv2.waitKey(0)
    return binary

# 分割字符
def segment_characters(binary_image):
    # 校正图像
    #corrected_image = correct_skew(binary_image)
    #cv2.imshow("校正后的车牌", corrected_image)
    #cv2.resize(corrected_image, (800, 600))
    #cv2.waitKey(0)

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
            characters.append(resized_char)
            start = i
    # 删除第一个和最后一个字符
    if len(characters) > 2:
        characters = characters[1:-1]

    # 对每个字符应用模板分割函数，将分割的字符图像去除边缘并居中
    processed_characters = []
    for char in characters:
        # 应用模板分割函数
        processed_char = template_segmentation(char)
        processed_characters.append(processed_char)
    return processed_characters


# 模板分割函数，对分割的字符图像进行裁剪和尺寸归一化处理
def template_segmentation(origin_img):
    THRESHOLD = 2 * 255
    # 提取字符各列满足条件(有两个255的单元格)的索引
    col_sums = np.sum(origin_img, axis=0)
    col_index = np.where(col_sums >= THRESHOLD)[0]
    # 提取字符各行满足条件(有两个255的单元格)的索引
    row_sums = np.sum(origin_img, axis=1)
    row_index = np.where(row_sums >= THRESHOLD)[0]
    # 检查索引是否为空
    if len(row_index) == 0 or len(col_index) == 0:
        raise ValueError("无法找到符合条件的行列，图像可能为空或不符合模板要求")
    # 提取字符区域
    output_img = origin_img[np.min(row_index):np.max(row_index)+1, np.min(col_index):np.max(col_index)+1]
    # 判断是否需要填充而不是缩放
    if col_index.shape[0] <= 3 or row_index.shape[0] <= 3:
        pad_row1 = int(np.floor((25 - output_img.shape[0]) / 2))
        pad_row2 = int(np.ceil((25 - output_img.shape[0]) / 2))
        pad_col1 = int(np.floor((15 - output_img.shape[1]) / 2))
        pad_col2 = int(np.ceil((15 - output_img.shape[1]) / 2))
        output_img = np.pad(output_img, ((pad_row1, pad_row2), (pad_col1, pad_col2)), 'constant', constant_values=(0, 0))
    else:
        output_img = cv2.resize(output_img, (15, 25), interpolation=0)
    return np.uint8(output_img)

# 生成模板数组
def template_array_generator(template_path, template_size):
    templates_array = np.zeros([template_size, 25, 15], dtype=np.uint8)
    index = 0
    files = os.listdir(template_path)
    for file in files:
        file_path = os.path.join(template_path, file)
        try:
            # 读取图像并解码
            img_data = np.fromfile(file_path, dtype=np.uint8)
            template_img = cv2.imdecode(img_data, -1)
            # 转换为灰度图
            template_img_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
            # 二值化处理
            _, template_img_binary = cv2.threshold(template_img_gray, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # 图像反转：模板与实际车牌相反
            template_img_binary = 255 - template_img_binary
            # 分割处理
            segmented = template_segmentation(template_img_binary)
            if segmented.shape != (25, 15):
                raise ValueError(f"模板分割结果尺寸不匹配: {file} -> {segmented.shape}")
            templates_array[index, :, :] = segmented
            index += 1
            if index >= template_size:
                break  # 达到指定数量后停止处理

        except Exception as e:
            print(f"[警告] 处理文件 {file} 时出错: {e}")
            continue
    if index == 0:
        raise ValueError("未成功加载任何模板图像，请检查模板路径和文件格式")

    return templates_array

# 识别字符
def recognize_character(char_img, templates, labels):
    # 初始化最小差异 min_diff 为无穷大，匹配字符 matched_char 为空
    min_diff = float('inf')
    matched_char = ''
    for i, template in enumerate(templates):
        # 遍历所有模板，计算当前模板与字符图像的像素差值绝对值总和
        diff = np.sum(np.abs(char_img.astype("int32") - template.astype("int32")))
        # 如果当前模板差异更小，则更新最小差异和匹配字符
        if diff < min_diff:
            min_diff = diff
            matched_char = labels[i]
    return matched_char

# 读取标签文件
def read_labels(label_file, encoding='gbk'):
    with open(label_file, 'r', encoding=encoding) as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    return labels

if __name__ == "__main__":
    # 加载车牌图像
    image = cv2.imread("images/ps1.png")
    image = cv2.resize(image, (800, 600))
    # 定位车牌
    plate = locate_license_plate(image)
    if plate is not None:
        # 预处理车牌
        binary_plate = preprocess_license_plate(plate)
        # 分割字符
        characters = segment_characters(binary_plate)
        # 使用matplotlib绘制7个子图
        fig, axs = plt.subplots(1, 7, figsize=(14, 2))  # 1行7列的子图布局
        # 确保只绘制最多7个字符图像
        for i in range(min(len(characters), 7)):
            axs[i].imshow(characters[i], cmap='gray')  # 灰度显示
            axs[i].axis('off')  # 关闭坐标轴
        # 对于剩余未使用的子图也关闭坐标轴以保持整洁（可选）
        for j in range(i + 1, 7):
            axs[j].axis('off')
        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
        plt.show()
        #
        # # 进行标签识别
        # chinese_labels = read_labels('carNumber/character.txt')
        # letter_labels = read_labels('carNumber/letter.txt')
        # number_labels = read_labels('carNumber/num.txt')
        # # 加载模板
        # chinese_templates = template_array_generator('carNumber/character', len(chinese_labels))
        # letter_templates = template_array_generator('carNumber/letter', len(letter_labels))
        # number_templates = template_array_generator('carNumber/num', len(number_labels))
        #
        # recognized_chars = []
        # for i, char in enumerate(characters):
        #     if i == 0:
        #         # 第一个有效字符：匹配汉字
        #         best_label = recognize_character(char, chinese_templates, chinese_labels)
        #     elif i == 1:
        #         # 第二个有效字符：只匹配字母
        #         best_label = recognize_character(char, letter_templates, letter_labels)
        #     else:
        #         # 合并字母和数字的模板与标签
        #         combined_templates = np.concatenate((letter_templates, number_templates), axis=0)
        #         combined_labels = letter_labels + number_labels
        #         best_label = recognize_character(char, combined_templates, combined_labels)
        #     recognized_chars.append(best_label)
        # print("识别结果:", ''.join(recognized_chars))



