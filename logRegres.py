import numpy as np
import matplotlib.pyplot as plt
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()  #strip()方法用于移除字符串头尾指定的字符
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat
def sigmoid(inX):   #sigmoid函数
    return 1.0 / (1 + np.exp(-inX))
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)  #将输入矩阵转换为numpy矩阵
    labelMat = np.mat(classLabels).transpose()  #将输入矩阵转换为numpy矩阵，并且将行向量转换为列向量
    m, n = np.shape(dataMatrix) #shape()返回矩阵的长度
    alpha = 0.001   #目标移动的步长
    maxCycles = 500 #迭代次数
    weights = np.ones((n, 1))
    for k in range(maxCycles):  #计算真实类别与预测类别的差值
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights
def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker = 's')
    ax.scatter(xcord2, ycord2, s = 30, c = 'green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
select = int(input("请输入你要选择的操作："))
if select == 1:
    dataArr, labelMat = loadDataSet()
    print(gradAscent(dataArr, labelMat))
elif select == 2:
    dataArr, labelMat = loadDataSet()
    weights = gradAscent(dataArr, labelMat)
    plotBestFit(weights.getA())