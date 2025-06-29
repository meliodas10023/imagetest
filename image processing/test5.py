import cv2
import numpy as np
import matplotlib.pyplot as plt
# 1. 车牌定位

def locate_license_plate(image):
    h, w = image.shape[:2]
    plate_mask = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            Rij, Gij, Bij = image[i, j, 2], image[i, j, 1], image[i, j, 0]
            if (Rij/Bij<0.35  and  Gij/Bij<0.9 and Bij>90 ) or (Gij/Bij< 0.35 and Rij/Bij<0.9 and Bij<90):
                plate_mask[i , j] = 255

    return plate_mask

# def locate_license_plate(image):
#     # 转换为HSV颜色空间
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     H = hsv[:, :, 0]
#     S = hsv[:, :, 1].astype(np.float32) / 255.0
#     V = hsv[:, :, 2].astype(np.float32) / 255.0
#
#     # HSV阈值掩码
#     mask = ((H >= 190) & (H <= 245) &
#             (S >= 0.35) & (S <= 1) &
#             (V >= 0.3) & (V <= 1)).astype(np.uint8)
#
#     # 行列投影分析
#     row_threshold = image.shape[1] * 0.1
#     col_threshold = image.shape[0] * 0.05
#
#     # 行投影
#     row_sums = np.sum(mask, axis=1)
#     rows = np.where(row_sums >= row_threshold)[0]
#     if len(rows) == 0:
#         return None
#     top, bottom = rows.min(), rows.max()
#
#     # 列投影
#     col_sums = np.sum(mask, axis=0)
#     cols = np.where(col_sums >= col_threshold)[0]
#     if len(cols) == 0:
#         return None
#     left, right = cols.min(), cols.max()
#
#     return image[top:bottom + 1, left:right + 1]

# 2.分割区域灰度化、二值化
def preprocess_plate(image):
    # 灰度化
    gray_plate = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # 二值化
    _, binary_plate = cv2.threshold(gray_plate,127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_plate



if __name__ == '__main__':
    image = cv2.imread('images/5-carNumber/5.jpg')
    cv2.imshow('original image',image)
    cv2.waitKey(0)
    plate_mask = locate_license_plate(image)
    cv2.imshow('plate mask',plate_mask)
    cv2.waitKey(0)
    binary_plate = preprocess_plate(plate_mask)
    cv2.imshow('binary plate',binary_plate)
    cv2.waitKey(0)





