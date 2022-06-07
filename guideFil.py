import cv2
import numpy as np


# 单通道的引导滤波实现，对照公式进行
def guidedFilter_oneChannel(srcImg, guidedImg, size=8, eps=0.01):
    """
    srcImg:引导图       #这里命名命反了，输入时先输入引导图再输入原图即可
    guidedImg:原图
    size:邻域大小
    eps:公式参数
    """
    # print(srcImg)
    # 像素值归一化
    srcImg = np.array(srcImg) / 255.0
    guidedImg = np.array(guidedImg) / 255.0
    wsize = (size, size)
    # img_shape = np.shape(srcImg)  # 获取图片大小
    srcemean = cv2.blur(srcImg, wsize)  # 对需处理的图片和引导图进行方框滤波
    guidedmean = cv2.blur(guidedImg, wsize)
    S_square_mean = cv2.blur(srcImg * srcImg, wsize)  # 对处理图与引导图之积和引导图平方进行方框滤波
    S_G_mean = cv2.blur(srcImg * guidedImg, wsize)
    var = S_square_mean - srcemean * srcemean  # 方差
    cov = S_G_mean - srcemean * guidedmean  # 协方差

    a = cov / (var + eps)  # 求取中间参数
    b = guidedmean - a * srcemean

    a_mean = cv2.blur(a, wsize)
    b_mean = cv2.blur(b, wsize)

    dstImg = a_mean * srcImg + b_mean
    dstImg = dstImg * 255.0  # 最后将像素值恢复
    return dstImg.astype(np.uint8)  # 转为uint


"""
def guidedFilter_oneChannel(I, g, wsize=8, eps=0.01):
    # print(I)
    winSize = (wsize, wsize)
    meanI = cv2.blur(I, winSize)
    meanG = cv2.blur(g, winSize)
    # print(meanI)
    meanII = cv2.blur(I * I, winSize)
    meanIG = cv2.blur(I * g, winSize)
    # print(meanII)
    # 方差
    varI = meanII - meanI * meanI
    # print("varI", varI)
    # 协方差
    covIG = meanIG - meanI * meanG

    a = covIG / (varI + eps)
    b = meanG - a * meanI
    meana = cv2.blur(a, winSize)
    meanb = cv2.blur(b, winSize)
    q = meana * I + meanb
    print("q", q)
    return q.astype(np.uint8)
"""

# 三通道引导滤波
def guidedFilter_threeChannel(srcImg, guidedImg, size=2, eps=0.01):

    img_shape = np.shape(srcImg)  # 获取图片大小
    dstImg = np.zeros(img_shape)
    for i in range(0, img_shape[2]):  # 对每一个通道进行单通道引导滤波
        dstImg[:, :, i] = guidedFilter_oneChannel(
            srcImg[:, :, i], guidedImg[:, :, i], size, eps
        )

    return dstImg.astype(np.uint8)


def FastguideFilter(I, g, wsize=8, eps=0.01, s=0.5):
    """
    快速引导滤波
    通过下采样减少像素点,计算mean_a & mean_b后进行上采样恢复到原有的尺寸大小。
    假设缩放比例为s,那么缩小后像素点的个数为N*s^2,那么时间复杂度由(N/s^2)变为O(N*s^2)
    """
    # 输入图像的高、宽
    h, w = I.shape[:2]
    I = np.array(I) / 255.0
    g = np.array(g) / 255.0
    # 缩小图像
    size = (int(round(w * s)), int(round(h * s)))

    small_I = cv2.resize(I, size, interpolation=cv2.INTER_CUBIC)
    small_G = cv2.resize(g, size, interpolation=cv2.INTER_CUBIC)

    small_winSize = (int(round(wsize * s)), int(round(wsize * s)))

    # 均值平滑
    mean_small_I = cv2.blur(small_I, small_winSize)
    mean_small_G = cv2.blur(small_G, small_winSize)

    # I*I和I*p的均值平滑
    mean_small_II = cv2.blur(small_I * small_I, small_winSize)

    mean_small_IG = cv2.blur(small_I * small_G, small_winSize)

    # 方差
    var_small_I = mean_small_II - mean_small_I * mean_small_I  # 方差公式

    # 协方差
    cov_small_IG = mean_small_IG - mean_small_I * mean_small_G

    small_a = cov_small_IG / (var_small_I + eps)
    small_b = mean_small_G - small_a * mean_small_I

    # 对a、b进行均值平滑
    mean_small_a = cv2.blur(small_a, small_winSize)
    mean_small_b = cv2.blur(small_b, small_winSize)

    # 放大
    size1 = (w, h)
    mean_a = cv2.resize(mean_small_a, size1, interpolation=cv2.INTER_LINEAR)
    mean_b = cv2.resize(mean_small_b, size1, interpolation=cv2.INTER_LINEAR)

    q = mean_a * I + mean_b
    q = q * 255.0
    return q.astype(np.uint8)


def main():
    img = cv2.imread("im1.bmp")
    print(np.shape(img))
    dstimg = guidedFilter_threeChannel(img, img, 8, 0.01)  # 以输入图为引导图的导向滤波
    dstimg1 = cv2.bilateralFilter(img, 8, sigmaColor=100, sigmaSpace=100)
    print(np.shape(dstimg))
    cv2.imwrite("output.jpg", dstimg)
    cv2.imshow("source", img)
    cv2.imshow("bilateral", dstimg1)
    cv2.imshow("output", dstimg)
    key = cv2.waitKey(0)
    if key == 32:
        cv2.destroyAllWindows()
        exit(0)


if __name__ == "__main__":
    main()
