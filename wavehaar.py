import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt
import math
import copy

np.seterr(divide="ignore", invalid="ignore")
"""
# 仅灰度图像格式转换,后发现np中已经有函数可以做
def float32_to_uint8(img):
    img1 = np.zeros(np.shape(img))
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            if img[i][j] > 255:
                img1[i][j] = 255
            elif img[i][j] < 0:
                img1[i][j] = 0
            else:
                img1[i][j] = math.floor(img[i][j])
    return img1
"""


def haar_img(img):
    threshold1 = 0.6
    if len(img.shape) == 2:
        img_f32 = img.astype(np.float32)
    else:
        img_f32 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 二维小波一级变换
    coeffs = pywt.dwt2(img_f32, "haar")
    cA, (cH, cV, cD) = coeffs
    # print(max(coeffs[1]))
    coeffs1 = list(coeffs)
    coeffs1[0] = copy.copy(coeffs[0])
    for i in range(1, len(coeffs)):
        coeffs1[i] = pywt.threshold(
            coeffs[i], threshold1 * np.max(coeffs[i]), mode="soft"
        )  # 将噪声滤波
    coeffs1 = tuple(coeffs1)
    datarec = pywt.idwt2(
        coeffs1,
        "haar",
    )  # 将信号进行滤波后的小波重构
    # 将各个子图进行拼接，最后得到一张图
    AH = np.concatenate([cA, cH], axis=1)
    VD = np.concatenate([cV, cD], axis=1)
    img = np.concatenate([AH, VD], axis=0)
    # plt.imshow(img1, "gray")
    # plt.show()
    coeffs = list(coeffs)
    # plt.imshow(img, "gray")
    # plt.savefig("edge11111111.png", bbox_inches="tight", pad_inches=0.0)
    # plt.show()
    return [datarec, img, coeffs]  # 分别为去噪图像，小波变换图像拼接（只用于显示），小波变换输出


def modulus_maxima(coeffs, threshold):

    modulus = np.sqrt(np.power(coeffs[1][0], 2) + np.power(coeffs[1][1], 2))
    atan = np.arctan(np.array(coeffs[1][1]) / np.array(coeffs[1][0]))
    atan_temp = np.zeros(np.shape(atan), "uint8")
    for i in range(len(atan)):  # 量化，先打上标签标注不同方向，对不同方向有不同的处理方法
        for j in range(len(atan[0])):
            if coeffs[1][1][i][j] > 0 and coeffs[1][0][i][j] > 0:
                atan_temp[i][j] = (
                    0
                    if (atan[i][j] < math.pi / 8)
                    else 1
                    if (atan[i][j] > math.pi / 8 and atan[i][j] < 3 * math.pi / 8)
                    else 2
                )
            elif coeffs[1][1][i][j] > 0 and coeffs[1][0][i][j] < 0:
                atan_temp[i][j] = (
                    2
                    if (atan[i][j] < -3 * math.pi / 8)
                    else 3
                    if (atan[i][j] > -3 * math.pi / 8 and atan[i][j] < -math.pi / 8)
                    else 4
                )
            elif coeffs[1][1][i][j] < 0 and coeffs[1][0][i][j] < 0:
                atan_temp[i][j] = (
                    4
                    if (atan[i][j] < math.pi / 8)
                    else 5
                    if (atan[i][j] > math.pi / 8 and atan[i][j] < 3 * math.pi / 8)
                    else 6
                )
            else:
                atan_temp[i][j] = (
                    6
                    if (atan[i][j] < -3 * math.pi / 8)
                    else 7
                    if (atan[i][j] > -3 * math.pi / 8 and atan[i][j] < -math.pi / 8)
                    else 0
                )

    for i in range(len(atan)):  # 求解模极大值，非模极大值置零
        for j in range(len(atan[0])):
            if atan_temp[i][j] == 6:  # 对8个标签
                if (
                    modulus[i][j]  # 若模值小于方向临近两点的值舍去置零
                    < modulus[min(i + 1, len(atan) - 1)][min(j + 1, len(atan[0]) - 1)]
                    or modulus[i][j] < modulus[min(i + 1, len(atan) - 1)][max(j - 1, 0)]
                ):
                    modulus[i][j] = 0
            elif atan_temp[i][j] == 1:
                if (
                    modulus[i][j] < modulus[max(i - 1, 0)][j]
                    or modulus[i][j] < modulus[i][min(j + 1, len(atan[0]) - 1)]
                ):
                    modulus[i][j] = 0
            elif atan_temp[i][j] == 2:
                if (
                    modulus[i][j] < modulus[max(i - 1, 0)][min(j + 1, len(atan[0]) - 1)]
                    or modulus[i][j] < modulus[max(i - 1, 0)][max(j - 1, 0)]
                ):
                    modulus[i][j] = 0
            elif atan_temp[i][j] == 3:
                if (
                    modulus[i][j] < modulus[max(i - 1, 0)][j]
                    or modulus[i][j] < modulus[i][max(j - 1, 0)]
                ):
                    modulus[i][j] = 0
            elif atan_temp[i][j] == 4:
                if (
                    modulus[i][j] < modulus[min(i + 1, len(atan) - 1)][max(j - 1, 0)]
                    or modulus[i][j] < modulus[max(i - 1, 0)][max(j - 1, 0)]
                ):
                    modulus[i][j] == 0
            elif atan_temp[i][j] == 5:
                if (
                    modulus[i][j] < modulus[i][max(j - 1, 0)]
                    or modulus[i][j] < modulus[min(i + 1, len(atan) - 1)][j]
                ):
                    modulus[i][j] = 0
            elif atan_temp[i][j] == 0:
                if (
                    modulus[i][j]
                    < modulus[min(i + 1, len(atan) - 1)][min(j + 1, len(atan[0]) - 1)]
                    or modulus[i][j]
                    < modulus[max(i - 1, 0)][min(j + 1, len(atan[0]) - 1)]
                ):
                    modulus[i][j] = 0
            else:
                if (
                    modulus[i][j] < modulus[i][min(j + 1, len(atan[0]) - 1)]
                    or modulus[i][j] < modulus[min(i + 1, len(atan) - 1)][j]
                ):
                    modulus[i][j] = 0
    # 设一些初值
    win = 7  # 自适应窗口为2*7+1 = 15
    temp = 0
    count = 0
    TN = 0
    T0 = np.average(modulus)

    for i in range(0, len(atan)):
        for j in range(0, len(atan[0])):
            # 这一步是使用固定阈值时使用的，默认使用自适应阈值，这里被注释掉
            # modulus[i][j] = 0 if (modulus[i][j] < 3.5 * T0) else 255

            temp = 0
            count = 0
            for k in range(max(0, i - win), min(len(atan), i + win)):
                for l in range(max(0, j - win), min(j + win, len(atan[0]))):
                    count += 1
                    temp += modulus[k][l]
            temp = temp / count
            TN = threshold * T0 + 0.5 * temp
            modulus[i][j] = 0 if (modulus[i][j] < TN) else 255

    return [modulus, atan_temp, atan]


def db2_img(img):
    threshold1 = 0.6
    if len(img.shape) == 2:
        img_f32 = img.astype(np.float32)
    else:
        img_f32 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # 二维小波一级变换
    coeffs = pywt.dwt2(img_f32, "db2")
    cA, (cH, cV, cD) = coeffs
    # print(max(coeffs[1]))
    coeffs1 = list(coeffs)
    coeffs1[0] = copy.copy(coeffs[0])
    for i in range(1, len(coeffs)):
        coeffs1[i] = pywt.threshold(
            coeffs[i], threshold1 * np.max(coeffs[i]), mode="soft"
        )  # 将噪声滤波，软阈值threshold1 * np.max(coeffs[i])
    coeffs1 = tuple(coeffs1)
    datarec = pywt.idwt2(
        coeffs1,
        "db2",
    )  # 将信号进行滤波后的小波重构

    # 将各个子图进行拼接，最后得到一张图
    AH = np.concatenate([cA, cH], axis=1)
    VD = np.concatenate([cV, cD], axis=1)
    img = np.concatenate([AH, VD], axis=0)
    # print(np.shape(img))
    # print(np.shape(datarec))
    # plt.imshow(img, "gray")
    # plt.savefig("edge11111111.png", bbox_inches="tight", pad_inches=0.0)
    # plt.show()
    coeffs = list(coeffs)
    return [datarec, img, coeffs]


""" 
def haar_denoise_img(img):
    img_u8 = np.copy(img)
    if len(img_u8.shape) == 2:
        img_f32 = img_u8.astype(np.float32)
    else:
        img_f32 = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 二维小波一级变换
    coeffs = pywt.dwt2(img_f32, "haar")
    cA, (cH, cV, cD) = coeffs
    cH1 = np.mean(np.absolute(cH))
    cV1 = np.mean(np.absolute(cV))
    print(cH)
    print(cH1)
    print(cV1)
    for i in range(np.shape(cA)[0]):
        for j in range(np.shape(cA)[1]):
            if math.fabs(cH[i][j]) > cH1:
                continue
            else:
                cH[i][j] = 0
    for i in range(np.shape(cA)[0]):
        for j in range(np.shape(cA)[1]):
            if math.fabs(cV[i][j]) > cV1:
                continue
            else:
                cV[i][j] = 0
    print(cH)
    # 将各个子图进行拼接，最后得到一张图
    AH = np.concatenate([cA, cH], axis=1)
    VD = np.concatenate([cV, cD], axis=1)
    img = np.concatenate([AH, VD], axis=0)
    return [[cA, cH, cV, cD], img]
 """

if __name__ == "__main__":
    img = cv2.imread(r"data\project2\lena.png")
    # img = haar_denoise_img(img)
    # print(img)
    # cv2.imshow("img", img)
    plt.imshow(img, "gray")
    plt.title("img")
    plt.show()
