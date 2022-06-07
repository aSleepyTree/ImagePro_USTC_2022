# ImagePro_USTC_2022
Image Processing, Department of Automation, USTC。2022

# 关于文件结构

----

1. `data`文件夹中为提供的数据
2. `img`文件夹中为实验的一些中间图像由于篇幅原因无法在文章中给出
3. `latex`文件夹中为文章的latex编辑文件
4. 以.py结尾的文件为实验中调用的python函数文件
5. test.py为主程序，其中的project1、project2以及duochidu函数实现了作业要求
6. report.pdf为报告文档
7. `noise.py`为图片添加噪声
8. `bilateralFli`实现双边滤波相关操作
9. `guidedFli.py`实现引导滤波相关操作
10. `wavehaar.py`实现基于哈尔小波和多贝西小波的小波变换操作


# 关于代码

----

实验用到的函数代码实现基本都是按照算法公式按步骤给出，相应函数中已给出尽可能详细的注释

关于多尺度的模极大值方法代码中以j=3为例，要实现更高的尺度只需将操作重复即可；由于第一次理解方法时理解出现失误导致认为代码不多没有进行分块管理后来按步骤推导时不想抛弃原来的代码致使代码没有将重复性工作函数化显得冗长。这里对代码做简单解释
```python
    img1 = wavehaar.haar_img(img_noise)  # 小波变化
    img11 = wavehaar.haar_img(np.array(img1[2][0]))  # 尺度+1的小波分解
    img111 = wavehaar.haar_img(np.array(img11[2][0]))  # 尺度+1的小波分解，至此尺度为3

```
实现三个尺度的小波变换
```python
    edge11, edge11_atan, _ = wavehaar.modulus_maxima(img111[2], 1)
    zeros111 = np.zeros(np.shape(edge11))  # 初始化一些全0的array备用
    zeros11 = np.zeros(np.shape(img11t[0]))
    zeros1 = np.zeros(np.shape(img1t[0]))
    count = 0  # 定义一些用到的变量
    count1 = count
    ii = 0
    jj = 0
```
第一次模极大值滤波同时定义一些变量
```python
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
```
这里实现了第一次的找足够长度的边缘的操作，对不够长的边缘使用flag = 1 重复一遍找边的路径并同时对相应的点赋0。其中judge函数实现了对边缘的追踪
```python
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
```
使用刚得到的对边缘选择过得大尺度图片对其非0点取3 $\times$ 3 的邻域保留其余点置零。

----
之后为上述过程的重复，直到最小尺度
