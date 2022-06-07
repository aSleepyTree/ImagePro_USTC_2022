import cv2
import numpy as np
import math
import copy

size = 5
delta = 10


def norm(img1, img2):
    """
    计算范数作为图像差别度量作为参考
    """
    x = cv2.absdiff(img1, img2)
    x_norm = np.linalg.norm(x, ord=None, axis=None, keepdims=False)
    return x_norm


def spilt(a):
    if a % 2 == 0:
        x1 = x2 = a / 2
    else:
        x1 = math.floor(a / 2)
        x2 = a - x1
    return int(-x1), int(x2)


# 打表，像素差值权重
def d_value(sigmaColor):
    value = np.zeros(256)
    for i in range(0, 255):
        t = i * i
        value[i] = math.e ** (-t / (2 * sigmaColor * sigmaColor))
    return value


# 打表，gauss权重
def gaussian_b0x(a, b, sigmaSpace):
    box = []
    x1, x2 = spilt(a)
    y1, y2 = spilt(b)
    for i in range(x1, x2):
        for j in range(y1, y2):
            t = i * i + j * j
            re = math.e ** (-t / (2 * sigmaSpace * sigmaSpace))
            box.append(re)
    # for x in box :
    #     print (x)
    return box


# 将(i,j)邻域内的点值写入列表，若越界则在相应的位置以(i,j)的值代替，最终返回列表的大小为a*b
def original(i, j, a, b, img):
    x1, x2 = spilt(a)
    y1, y2 = spilt(b)
    temp = np.zeros(a * b)  # temp即为邻域内的点值
    count = 0
    for m in range(x1, x2):
        for n in range(y1, y2):
            if (  # 越界
                i + m < 0
                or i + m > img.shape[0] - 1
                or j + n < 0
                or j + n > img.shape[1] - 1
            ):
                temp[count] = img[i, j]
            else:
                temp[count] = img[i + m, j + n]
            count += 1
    return temp


def bilateral_function(a, b, img, gauss_fun, d_value_e):
    x1, x2 = spilt(a)
    y1, y2 = spilt(b)
    re = np.zeros(a * b)
    img0 = copy.copy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if j == 0 and i % 4 == 0:
                print(i, "/", img.shape[0])
            temp = original(i, j, a, b, img0)
            if j == 270 and i == 270:
                print(temp)  # 显示进度
            # print("ave:",ave_temp)
            count = 0
            for m in range(x1, x2):
                for n in range(y1, y2):
                    # print(m, n)
                    # print(img.shape[0], img.shape[1])
                    if (
                        i + m < 0
                        or i + m > img.shape[0] - 1
                        or j + n < 0
                        or j + n > img.shape[1] - 1
                    ):
                        x = img[i, j]
                    else:
                        x = img[i + m, j + n]
                    t = int(math.fabs(int(x) - int(img[i, j])))  # 像素差
                    re[count] = d_value_e[t]  # 计算权重，查表
                    count += 1
            evalue = np.multiply(re, gauss_fun)  # 像素值权重与gauss权重相乘
            img[i, j] = int(np.average(temp, weights=evalue))  # 加权平均
    print("==================")
    return img


def bilateral(img, size=12, sigmaColor=1, sigmaSpace=1):
    gauss = gaussian_b0x(size, size, sigmaSpace)  # 读取权重表
    d_value_v = d_value(sigmaColor)  # 读取权重表
    re = bilateral_function(size, size, copy.copy(img), gauss, d_value_v)
    return re


def main():  # main函数均为测试所用
    gauss = gaussian_b0x(size, size)  # 读取权重表
    d_value_v = d_value()  # 读取权重表
    img0 = cv2.imread(r"data\project1\im1.bmp")
    bilateral_img = bilateral_function(size, size, copy.copy(img0), gauss, d_value_v)
    # print(norm(bilateral_img, img0))
    cv2.imshow("shuangbian", bilateral_img)
    cv2.imshow("yuantu", img0)
    cv2.imwrite("shuangbian.jpg", bilateral_img)
    key = cv2.waitKey(0)
    if key == 32:
        cv2.destroyAllWindows()
        exit(0)


if __name__ == "__main__":
    main()
