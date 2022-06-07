from sys import modules
import cv2
import numpy as np
import bilateralFil as bf
import guideFil as gf
import time
import noise
import wavehaar
import matplotlib.pyplot as plt


size = 12
size1 = 2

"""
所有作业相关已经放到project1()完成第一大问相关,project2()完成第二大问除
多尺度相关和duochidu()完成多尺度问题中,检测时将运行对应函数即可,第二个大
问题的模极大值法相关需要需要将特定部分代码注释取消并注释其他相关代码,相应
位置有注释解释。最后面注释掉的代码均为早些时候测试所用,不关心即可
"""


def judge(edge11, edge11_atan, zeros111, i, j, count, flag):
    # judge函数用于寻找从（i，j）开始的单像素的边缘连线并返回下一个点的位置及目前的线长
    count += 1
    if edge11_atan[i][j] == 0:  # 对8个标签分别操作，0即正右方，剩下的按逆时针即1指右上方等等
        if edge11[i][min(j + 1, len(edge11_atan[0]) - 1)]:
            zeros111[i][j] = 1
            zeros111[i][min(j + 1, len(edge11_atan[0]) - 1)] = 1
        edge11[i][j] = 0 if flag == 1 else edge11[i][j]
        return (  # 若到达边界返回0，否则返回下一个点的坐标和当前边缘线长度
            [0, 0, 0]
            if min(j + 1, len(edge11_atan[0]) - 1) == j
            else [i, min(j + 1, len(edge11_atan[0]) - 1), count]
        )
    elif edge11_atan[i][j] == 1:
        if edge11[max(i - 1, 0)][min(j + 1, len(edge11_atan[0]) - 1)]:
            zeros111[i][j] = 1
            zeros111[max(i - 1, 0)][min(j + 1, len(edge11_atan[0]) - 1)] = 1

        edge11[i][j] = 0 if flag == 1 else edge11[i][j]
        return (
            [0, 0, 0]
            if min(j + 1, len(edge11_atan[0]) - 1) == j and max(i - 1, 0) == i
            else [max(i - 1, 0), min(j + 1, len(edge11_atan[0]) - 1), count]
        )

    elif edge11_atan[i][j] == 2:
        if edge11[max(i - 1, 0)][j]:
            zeros111[i][j] = 1
            zeros111[max(i - 1, 0)][j] = 1

        edge11[i][j] = 0 if flag == 1 else edge11[i][j]
        return [0, 0, 0] if max(i - 1, 0) == i else [max(i - 1, 0), j, count]

    elif edge11_atan[i][j] == 3:
        if edge11[max(i - 1, 0)][max(j - 1, 0)]:
            zeros111[i][j] = 1
            zeros111[max(i - 1, 0)][max(j - 1, 0)] = 1

        edge11[i][j] = 0 if flag == 1 else edge11[i][j]
        return (
            [0, 0, 0]
            if max(i - 1, 0) == i and max(j - 1, 0) == j
            else [max(i - 1, 0), max(j - 1, 0), count]
        )

    elif edge11_atan[i][j] == 4:
        if edge11[i][max(j - 1, 0)]:
            zeros111[i][j] = 1
            zeros111[i][max(j - 1, 0)] = 1

        edge11[i][j] = 0 if flag == 1 else edge11[i][j]
        return [0, 0, 0] if max(j - 1, 0) == j else [i, max(j - 1, 0), count]

    elif edge11_atan[i][j] == 5:
        if edge11[min(i + 1, len(edge11_atan) - 1)][max(j - 1, 0)]:
            zeros111[i][j] = 1
            zeros111[min(i + 1, len(edge11_atan) - 1)][max(j - 1, 0)] = 1

        edge11[i][j] = 0 if flag == 1 else edge11[i][j]
        return (
            [0, 0, 0]
            if min(i + 1, len(edge11_atan) - 1) == i and max(j - 1, 0) == j
            else [min(i + 1, len(edge11_atan) - 1), max(j - 1, 0), count]
        )

    elif edge11_atan[i][j] == 6:
        if edge11[min(i + 1, len(edge11_atan) - 1)][j]:
            zeros111[i][j] = 1
            zeros111[min(i + 1, len(edge11_atan) - 1)][j] = 1

        edge11[i][j] = 0 if flag == 1 else edge11[i][j]
        return (
            [0, 0, 0]
            if min(i + 1, len(edge11_atan) - 1) == i
            else [min(i + 1, len(edge11_atan) - 1), j, count]
        )

    elif edge11_atan[i][j] == 7:
        if edge11[min(i + 1, len(edge11_atan) - 1)][
            min(j + 1, len(edge11_atan[0]) - 1)
        ]:
            zeros111[i][j] = 1
            zeros111[min(i + 1, len(edge11_atan) - 1)][
                min(j + 1, len(edge11_atan[0]) - 1)
            ] = 1

        edge11[i][j] = 0 if flag == 1 else edge11[i][j]
        return (
            [0, 0, 0]
            if (
                min(i + 1, len(edge11_atan) - 1) == i
                and j == min(j + 1, len(edge11_atan[0]) - 1)
            )
            else [
                min(i + 1, len(edge11_atan) - 1),
                min(j + 1, len(edge11_atan[0]) - 1),
                count,
            ]
        )
    edge11[i][j] = 0 if flag == 1 else edge11[i][j]
    return [0, 0, 0]


def project1():
    img = cv2.imread(r"data\project1\im3.bmp", 0)  # 读取噪声图以及引导图的灰度图像
    imgg = cv2.imread(r"data\project1\im2.bmp", 0)
    bilateral = bf.bilateral(img, size, sigmaColor=1, sigmaSpace=1)
    bilateral1 = bf.bilateral(img, size, sigmaColor=1, sigmaSpace=10)
    bilateral2 = bf.bilateral(img, size, sigmaColor=1, sigmaSpace=100)
    bilateral3 = bf.bilateral(img, size, sigmaColor=10, sigmaSpace=1)
    bilateral4 = bf.bilateral(img, size, sigmaColor=10, sigmaSpace=10)
    bilateral5 = bf.bilateral(img, size, sigmaColor=10, sigmaSpace=100)
    bilateral6 = bf.bilateral(img, size, sigmaColor=100, sigmaSpace=1)
    bilateral7 = bf.bilateral(img, size, sigmaColor=100, sigmaSpace=10)
    bilateral8 = bf.bilateral(img, size, sigmaColor=100, sigmaSpace=100)
    cv2.imshow("bilateral_img", bilateral)
    # bilateral44 = bf.bilateral(imgg, size, sigmaColor=10, sigmaSpace=10)
    # cv2.imwrite("bilateral", bilateral44)
    key = cv2.waitKey(0)
    if key == 32:
        cv2.destroyAllWindows()
    """#这里是为了将不同的双标滤波用plt显示在一张图上
    bilateral = cv2.cvtColor(bilateral, cv2.COLOR_GRAY2BGR)  
    bilateral2 = cv2.cvtColor(bilateral2, cv2.COLOR_GRAY2BGR)
    bilateral3 = cv2.cvtColor(bilateral3, cv2.COLOR_GRAY2BGR)
    bilateral1 = cv2.cvtColor(bilateral1, cv2.COLOR_GRAY2BGR)
    bilateral4 = cv2.cvtColor(bilateral4, cv2.COLOR_GRAY2BGR)
    bilateral5 = cv2.cvtColor(bilateral5, cv2.COLOR_GRAY2BGR)
    bilateral6 = cv2.cvtColor(bilateral6, cv2.COLOR_GRAY2BGR)
    bilateral7 = cv2.cvtColor(bilateral7, cv2.COLOR_GRAY2BGR)
    bilateral8 = cv2.cvtColor(bilateral8, cv2.COLOR_GRAY2BGR)

    title = [
        "SigmaColor=1 SigmaSpace=1",
        "1_10",
        "1_100",
        "10_1",
        "10_10",
        "10_100",
        "100_1",
        "100_10",
        "100_100",
    ]
    bil = [
        bilateral,
        bilateral1,
        bilateral2,
        bilateral3,
        bilateral4,
        bilateral5,
        bilateral6,
        bilateral7,
        bilateral8,
    ]
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(bil[i])
        plt.title(title[i])
        plt.axis("off")
    plt.savefig("bilateralExample.png")
    plt.suptitle("bilateralExample")
    plt.show()

    key = cv2.waitKey(0)
    if key == 32:
        cv2.destroyAllWindows()
    """
    time0 = time.time()
    guided1 = gf.guidedFilter_oneChannel(imgg, img, size1, 0.04)  # 图3以图2为引导
    time1 = time.time()
    guided2 = gf.FastguideFilter(imgg, img, size1, 0.04)  # 快速引导滤波
    time2 = time.time()

    guided3 = gf.guidedFilter_oneChannel(img, img, size1, 0.04)  # 图3以图3为引导

    print("引导滤波用时", time1 - time0, "秒")
    print("快速引导滤波用时", time2 - time1, "秒")
    print(bf.norm(guided1, guided3))
    print(bf.norm(guided1, img))
    print(bf.norm(guided3, img))
    cv2.imshow("guidedFilter_oneChannel", guided1)  # 显示快速引导滤波，两种引导滤波以及双边滤波作比较
    cv2.imshow("32FastguideFilter", guided2)
    cv2.imshow("33guidedFilter_oneChannel", guided3)
    cv2.imshow("bilateral", bilateral4)
    cv2.imwrite("bil.png", bilateral4)
    cv2.imwrite("guided32.png", guided1)
    cv2.imwrite("guided33.png", guided3)
    cv2.imwrite("fastguided.png", guided2)
    key = cv2.waitKey(0)
    if key == 32:
        exit(0)
    return 0


def duochidu():
    img = cv2.imread(r"data\project2\lena.png", 0)  # 读取灰度图像
    img_noise = noise.add_gaussian_noise(img, 5)
    img1 = wavehaar.haar_img(img_noise)  # 小波变化
    img11 = wavehaar.haar_img(np.array(img1[2][0]))  # 尺度+1的小波分解
    img111 = wavehaar.haar_img(np.array(img11[2][0]))  # 尺度+1的小波分解，至此尺度为3
    img1t = [
        np.array(img1[2][0]),
        np.array(img1[2][1]),
    ]  # 将img1[2]元组转为array([],[[],[],[]])便于操作
    img11t = [
        np.array(img11[2][0]),
        np.array(img11[2][1]),
    ]  # 将img11[2]元组转为array([],[[],[],[]])
    edge11, edge11_atan, _ = wavehaar.modulus_maxima(img111[2], 1)
    zeros111 = np.zeros(np.shape(edge11))  # 初始化一些全0的array备用
    zeros11 = np.zeros(np.shape(img11t[0]))
    zeros1 = np.zeros(np.shape(img1t[0]))
    count = 0  # 定义一些用到的变量
    count1 = count
    ii = 0
    jj = 0
    for i in range(0, len(edge11)):
        for j in range(0, len(edge11[0])):
            count = 0
            ii = i
            jj = j
            if edge11[i][j] and not zeros111[i][j]:
                _, _, count = judge(edge11, edge11_atan, zeros111, i, j, count, 0)
                while count:  # 找边
                    count1 = count
                    i, j, count = judge(edge11, edge11_atan, zeros111, i, j, count, 0)
                    # print(i, j, count)
                    if count > 100:  # 边太长显然是陷入了相邻的两个点互相为梯度方向的循环导致连线函数一直在这两点间徘徊，这里强制跳出
                        break
                if count1 < 30:  # 如果边短，按照刚才走的路再走一遍，但flag = 1即走到的点都置零
                    while count1:  # 这里count1只有0和非0，其他数字已无意义
                        ii, jj, count1 = judge(
                            edge11, edge11_atan, zeros111, ii, jj, count1, 1
                        )
                        if (
                            count1 > 100
                        ):  # 边太长显然是陷入了相邻的两个点互相为梯度方向的循环导致连线函数一直在这两点间徘徊，这里强制跳出
                            break

    for i in range(0, len(edge11)):
        for j in range(0, len(edge11[0])):
            if edge11[i][j]:  # 边缘3*3邻域保留，对相应的点进行标记
                for l in range(max(2 * i - 1, 0), min(2 * i + 2, len(zeros11))):
                    for m in range(max(2 * j - 1, 0), min(2 * j + 2, len(zeros11[0]))):
                        zeros11[l][m] = 1
    for i in range(0, len(img11t[1][0])):  # 不在边缘（即无标记）周围的点进行清除
        for j in range(0, len(img11t[1][1][0])):
            if zeros11[i][j] == 0:
                img11t[1][0][i][j] = 0
                img11t[1][1][i][j] = 0
    zeros11 -= zeros11  # 将矩阵回0
    edge1, edge1_atan, _ = wavehaar.modulus_maxima(img11t, 1)  # 模极大值
    for i in range(0, len(edge1)):
        for j in range(0, len(edge1[0])):
            count = 0
            ii = i
            jj = j
            if edge1[i][j] and not zeros11[i][j]:
                i, j, count = judge(edge1, edge1_atan, zeros11, i, j, count, 0)
                while count:  # 找边
                    count1 = count
                    i, j, count = judge(edge1, edge1_atan, zeros11, i, j, count, 0)
                    # print(i, j, count)
                    if count > 100:  # 边太长显然是陷入了相邻的两个点互相为梯度方向的循环导致连线函数一直在这两点间徘徊，这里强制跳出
                        break
                if count1 < 30:  # 如果边短，按照刚才走的路再走一遍，但flag = 1即走到的点都置零
                    while count1:  # 这里count1只有0和非0，其他数字已无意义
                        ii, jj, count1 = judge(
                            edge1, edge1_atan, zeros11, ii, jj, count, 1
                        )
                        if (
                            count1 > 100
                        ):  # 边太长显然是陷入了相邻的两个点互相为梯度方向的循环导致连线函数一直在这两点间徘徊，这里强制跳出
                            break
    for i in range(0, len(edge1)):
        for j in range(0, len(edge1[0])):
            if edge1[i][j]:  # 边缘3*3邻域保留
                for l in range(max(2 * i - 1, 0), min(2 * i + 2, len(zeros1))):
                    for m in range(max(2 * j - 1, 0), min(2 * j + 2, len(zeros1[0]))):
                        zeros1[l][m] = 1
    for i in range(0, len(img1t[1][0])):  # 不在边缘周围的点进行清除
        for j in range(0, len(img1t[1][1][0])):
            if zeros1[i][j] == 0:
                img1t[1][0][i][j] = 0
                img1t[1][1][i][j] = 0
    edge, _, _ = wavehaar.modulus_maxima(img1t, 1)  # 到最低尺度，大尺度下的信息已经全部获得，再进行一次模极大值滤波即可
    plt.imshow(edge, "gray")
    plt.axis("off")
    plt.savefig("multi_dim_edge1.png", bbox_inches="tight", pad_inches=0.0)
    plt.show()
    return 0


def project2():
    img = cv2.imread(r"data\project2\lena.png", 0)  # 读取灰度图像
    img_noise = noise.add_gaussian_noise(img, 5)  # 将标准正态分布的5倍像素值作为噪声加入
    # cv2.imshow("noise", img_noise)
    # cv2.imshow("source img", img)
    # haar_img和db2_img可以实现边缘检测和去噪两种操作，下四行各自取去噪结果和边缘检测结果
    img1 = wavehaar.haar_img(img_noise)  # 去噪
    img2 = wavehaar.db2_img(img_noise)  # 去噪
    img3 = wavehaar.haar_img(img)  # 无噪声的边缘检测
    img4 = wavehaar.db2_img(img)  # 无噪声边缘检测
    """                                         #模极大值法，使用时取消注释并注释掉本函数后续部分
    edge,_,_ = wavehaar.modulus_maxima(img1[2],3)   # 对有噪声的图片进行模极大值法的边缘检测
    plt.imshow(edge, "gray")
    plt.axis("off")
    plt.savefig("edge.png", bbox_inches="tight", pad_inches=0.0)
    plt.show()
    exit(0)

    """

    cv2.imshow(
        "haar denoise img", img1[0].astype(np.uint8)
    )  # 去噪后的图像img[0]为去噪图像，img[1]为小波变化平均图加细节图拼接，img[2]为小波变化的直接输出
    cv2.imshow("db2 denoise img", img2[0].astype(np.uint8))
    cv2.imwrite("source.png", img)
    cv2.imwrite("noise.png", img_noise)
    cv2.imwrite("haar.png", img1[0].astype(np.uint8))
    cv2.imwrite("db2.png", img2[0].astype(np.uint8))

    plt.imshow(img3[1], "gray")  # 展示边缘检测
    plt.title("haar_edge")
    plt.axis("off")
    plt.savefig("haar_edge.png", bbox_inches="tight", pad_inches=0.0)
    plt.show()

    plt.imshow(img4[1], "gray")  # 展示边缘检测
    plt.title("db2_edge")
    plt.axis("off")
    plt.savefig("db2_edge.png", bbox_inches="tight", pad_inches=0.0)
    plt.show()
    key = cv2.waitKey(0)
    if key == 32:
        exit(0)
    return 0


# project1()
# exit(0)

# project2()
# exit(0)
duochidu()
exit(0)


'''
img2 = wavehaar.haar_denoise_img(img)
img3 = wavehaar.haar_img(img)
img4 = np.concatenate([img1, img2], axis=1)
img5 = np.concatenate([img4, img3], axis=0)
# print(img1)
# print(img_noise)
cv2.imshow("noise", img_noise)
cv2.imshow("source", img)
plt.imshow(img5, "gray")
plt.show()

cv2.waitKey(0)
exit(0)
bilateral = cv2.bilateralFilter(img, size, sigmaColor=100, sigmaSpace=100)
guided = cv2.ximgproc.guidedFilter(img, img, size, 0.16, -1)
time1 = time.time()
guided1 = gf.guidedFilter_oneChannel(img, img, size, 0.04)
guided2 = gf.guidedFilter_oneChannel(img, img, size, 0.16)
time2 = time.time()
guided3 = gf.FastguideFilter(img, img, size, 0.04, 0.5)
guided4 = gf.FastguideFilter(img, img, size, 0.16, 0.5)
time3 = time.time()
"""
print(type(guided[0][0]))
print(type(guided1[0][0]))
print(guided1)
print(guided)
"""
# cv2.imshow("bilateralFilter", bilateral)
cv2.imshow("1guidedFilter", guided1)
cv2.imshow("2guidedFilter", guided2)
cv2.imshow("3guidedFilter", guided3)
cv2.imshow("4guidedFilter", guided4)
print(time2 - time1, time3 - time2)
print(bf.norm(img, guided1))
print(bf.norm(img, guided2))
print(bf.norm(img, guided3))
print(bf.norm(img, guided4))
# 显示输出图像及 原图+细节（展示了梯度反转现象，细节=原图-双边滤波）
# cv2.imshow("resource + resource - bilateralFilter", img + img - bilateral)
"""
# cv2.imwrite("gradient reverse.jpg", img + img - bilateral)
# temp = np.array(img-bilateral)
# cv2.imshow('im4',temp)
# cv2.imwrite("bilateralFilter_col_10_sap_10.jpg", bilateral)

# bilateral = cv2.bilateralFilter(img, size, sigmaColor=1000, sigmaSpace=10)
# cv2.imshow("im3", bilateral)
# print(bf.norm(img, bilateral))
# cv2.imwrite("bilateralFilter_col_1000_sap_10.jpg", bilateral)

# bilateral = cv2.bilateralFilter(img, size, sigmaColor=10, sigmaSpace=1000)
# cv2.imshow("im4", bilateral)
# print(bf.norm(img, bilateral))
# cv2.imwrite("bilateralFilter_col_10_sap_1000.jpg", bilateral)

# bilateral = cv2.bilateralFilter(img, size, sigmaColor=1000, sigmaSpace=1000)
# cv2.imshow("im5", bilateral)
# print(bf.norm(img, bilateral))
# cv2.imwrite("bilateralFilter_col_1000_sap_1000.jpg", bilateral)
"""
key = cv2.waitKey(0)
if key == 32:
    cv2.destroyAllWindows()
    exit(0)
'''
