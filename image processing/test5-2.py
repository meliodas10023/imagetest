import os
import cv2
import numpy as np

img = cv2.imread('images/ps1.png')   # 最终用于识别的图像


# 1.车牌定位
def license_region(image):
    r = image[:, :, 2]
    g = image[:, :, 1]
    b = image[:, :, 0]
    # 求出三种阈值
    license_region_thresh = np.zeros(np.append(3, r.shape))    # 创建一个空的三维数组用于存放三种阈值
    license_region_thresh[0, :, :] = r/b
    license_region_thresh[1, :, :] = g/b
    license_region_thresh[2, :, :] = b
    # 存放满足阈值条件的像素点坐标
    region_origin = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (license_region_thresh[0, i, j] < 0.35 and
                license_region_thresh[1, i, j] < 0.9 and
                license_region_thresh[2, i, j] > 90) or (
                    license_region_thresh[1, i, j] < 0.35 and
                    license_region_thresh[0, i, j] < 0.9 and
                    license_region_thresh[2, i, j] < 90):
                region_origin.append([i, j])
    region_origin = np.array(region_origin)
    # 进一步缩小行的索引范围
    row_index = np.unique(region_origin[:, 0])
    row_index_number = np.zeros(row_index.shape, dtype=np.uint8)
    for i in range(region_origin.shape[0]):
        for j in range(row_index.shape[0]):
            if region_origin[i, 0] == row_index[j]:
                row_index_number[j] = row_index_number[j]+1
    row_index_out = row_index_number > 10   # 将误判的点去除
    row_index_out = row_index[row_index_out]
    # 进一步缩小列的索引范围
    col_index = np.unique(region_origin[:, 1])
    col_index_number = np.zeros(col_index.shape, dtype=np.uint8)
    for i in range(region_origin.shape[0]):
        for j in range(col_index.shape[0]):
            if region_origin[i, 1] == col_index[j]:
                col_index_number[j] = col_index_number[j]+1
    col_index_out = col_index_number > 10
    col_index_out = col_index[col_index_out]
    # 得出最后的区间
    region_out = np.array([[np.min(row_index_out), np.max(row_index_out)],
                           [np.min(col_index_out), np.max(col_index_out)]])
    return region_out


region = license_region(img)
# 显示车牌区域
img_test = img.copy()   # 拷贝时不能直接等号赋值
cv2.rectangle(img_test, pt1=(region[1, 0], region[0, 0]), pt2=(region[1, 1], region[0, 1]),
              color=(0, 0, 255), thickness=2)
cv2.imshow('car_license_region', img_test)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2.分割区域灰度化、二值化
img_car_license = img[region[0, 0]:region[0, 1], region[1, 0]:region[1, 1], :]
img_car_license_gray = cv2.cvtColor(img_car_license, cv2.COLOR_BGR2GRAY)    # 将RGB图像转化为灰度图像
# otus二值化
img_car_license_binary = cv2.threshold(img_car_license_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# 3.车牌分割（均分割为25*15的图片）height=25,width=15
# 模板分割函数，只针对单个字符，用于去除其周围的边缘，并resize
def template_segmentation(origin_img):
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


# 对原始车牌抠图，抠出每一个字符
temp_col_index = []
for col in range(img_car_license_binary.shape[1]):
    if np.sum(img_car_license_binary[:, col]) >= 2*255:     # 提取大于等于2个255的列
        temp_col_index.append(col)
temp_col_index = np.array(temp_col_index)
flag = 0    # 值是7个字符的起始列
flag_i = 0  # 值的变化范围：从0到6(对应车牌的7个字符)
car_license_out_col = np.uint8(np.zeros([7, 30]))   # 7行的数组存储车牌上的7个需识别的字
for j in range(temp_col_index.shape[0]-1):
    if temp_col_index[j+1]-temp_col_index[j] >= 2:   # 提取的>=2个255的列之间不是相邻的(可初步解决川的分割问题)
        temp = temp_col_index[flag:j+1]
        temp = np.append(temp, np.zeros(30-temp.shape[0]))  # 补成30维的向量，方便最后赋值给car_license_out_col
        temp = np.uint8(temp.reshape(1, 30))
        car_license_out_col[flag_i, :] = temp
        flag = j+1
        flag_i = flag_i+1
temp = temp_col_index[flag:]
temp = np.append(temp, np.zeros(30-temp.shape[0]))
temp = np.uint8(temp.reshape(1, 30))
car_license_out_col[flag_i, :] = temp

# 分别提取7个字符
car_license_out_row = np.uint8(np.zeros([7, 30]))
for row in range(car_license_out_row.shape[0]):    # car_license_out_row.shape[0]
    temp = car_license_out_col[row, :]
    index = 0
    for i in range(temp.shape[0]):  # 去除列索引中多余的0
        if temp[i] == 0:
            index = i
            break
    col_temp = temp[0:index]
    temp_img = img_car_license_binary[:, np.min(col_temp):np.max(col_temp)+1]
    t = np.nonzero(np.sum(temp_img, axis=1))
    if row == 0:
        province1 = temp_img[t, :]      # 汉字后续扩展成40*40
        province1 = province1[0, :, :]
        province1 = template_segmentation(province1)
        province1 = np.uint8(province1)
    if row == 1:
        province2 = temp_img[t, :]      # 字母和数字后续扩展成40*40
        province2 = province2[0, :, :]
        province2 = template_segmentation(province2)
        province2 = np.uint8(province2)
    if row == 2:
        car_number1 = temp_img[t, :]
        car_number1 = car_number1[0, :, :]
        car_number1 = template_segmentation(car_number1)
        car_number1 = np.uint8(car_number1)
    if row == 3:
        car_number2 = temp_img[t, :]
        car_number2 = car_number2[0, :, :]
        car_number2 = template_segmentation(car_number2)
        car_number2 = np.uint8(car_number2)
    if row == 4:
        car_number3 = temp_img[t, :]
        car_number3 = car_number3[0, :, :]
        car_number3 = template_segmentation(car_number3)
        car_number3 = np.uint8(car_number3)
    if row == 5:
        car_number4 = temp_img[t, :]
        car_number4 = car_number4[0, :, :]
        car_number4 = template_segmentation(car_number4)
        car_number4 = np.uint8(car_number4)
    if row == 6:
        car_number5 = temp_img[t, :]
        car_number5 = car_number5[0, :, :]
        car_number5 = template_segmentation(car_number5)
        car_number5 = np.uint8(car_number5)

cv2.imshow('province1', province1)
cv2.imshow('province2', province2)
cv2.imshow('car_number1', car_number1)
cv2.imshow('car_number2', car_number2)
cv2.imshow('car_number3', car_number3)
cv2.imshow('car_number4', car_number4)
cv2.imshow('car_number5', car_number5)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 4.车牌识别
# 读取原始图片并生成模板的函数
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


# 读取所有的汉字并生成模板
Chinese_character = open('images/5-carNumber/character.txt', encoding="gbk").read()
Chinese_character = Chinese_character.split("\n")
Chinese_char_template = template_array_generator('images/5-carNumber/character', len(Chinese_character))
# 读取所有的数字并生成模板
Number_character = open('images/5-carNumber/num.txt', encoding="gbk").read()
Number_character = Number_character.split("\n")
Number_char_template = template_array_generator('images/5-carNumber/num', len(Number_character))
# 读取所有的字母并生成模板
Alphabet_character = open('images/5-carNumber/letter.txt', encoding="gbk").read()
Alphabet_character = Alphabet_character.split("\n")
Alphabet_char_template = template_array_generator('images/5-carNumber/letter', len(Alphabet_character))

# 进行字符识别
car_character = np.uint8(np.zeros([7, 25, 15]))
car_character[0, :, :] = province1.copy()
car_character[1, :, :] = province2.copy()
car_character[2, :, :] = car_number1.copy()
car_character[3, :, :] = car_number2.copy()
car_character[4, :, :] = car_number3.copy()
car_character[5, :, :] = car_number4.copy()
car_character[6, :, :] = car_number5.copy()
match_length = Chinese_char_template.shape[0]+Alphabet_char_template.shape[0]+Number_char_template.shape[0]
match_mark = np.zeros([7, match_length])
Chinese_char_start = 0
Chinese_char_end = Chinese_char_template.shape[0]
Alphabet_char_start = Chinese_char_template.shape[0]
Alphabet_char_end = Chinese_char_template.shape[0]+Alphabet_char_template.shape[0]
Number_char_start = Chinese_char_template.shape[0]+Alphabet_char_template.shape[0]
Number_char_end = match_length
for i in range(match_mark.shape[0]):    # 7个需识别的字符
    for j in range(Chinese_char_start, Chinese_char_end):  # 所有的汉字模板
        match_mark[i, j] = cv2.matchTemplate(car_character[i, :, :], Chinese_char_template[j, :, :], cv2.TM_CCOEFF)
    # 所有的字母模板
    for j in range(Alphabet_char_start, Alphabet_char_end):
        match_mark[i, j] = cv2.matchTemplate(car_character[i, :, :],
                                             Alphabet_char_template[j-Alphabet_char_start, :, :],
                                             cv2.TM_CCOEFF)
    # 所有的数字模板
    for j in range(Number_char_start, Number_char_end):
        match_mark[i, j] = cv2.matchTemplate(car_character[i, :, :],
                                             Number_char_template[j-Number_char_start, :, :],
                                             cv2.TM_CCOEFF)
output_index = np.argmax(match_mark, axis=1)
output_char = []
for i in range(output_index.shape[0]):
    if 0 <= output_index[i] <= 28:
        output_char.append(Chinese_character[output_index[i]])
    if 29 <= output_index[i] <= 54:
        output_char.append(Alphabet_character[output_index[i]-29])
    if 55 <= output_index[i] <= 64:
        output_char.append(Number_character[output_index[i]-55])

# 打印识别结果
for i in range(len(output_char)):
    if i == 0:
        print('province1:'+output_char[0])
    if i == 1:
        print('province1:'+output_char[1])
    if i == 2:
        print('car1:'+output_char[2])
    if i == 3:
        print('car2:' + output_char[3])
    if i == 4:
        print('car3:' + output_char[4])
    if i == 5:
        print('car4:' + output_char[5])
    if i == 6:
        print('car5:' + output_char[6])
