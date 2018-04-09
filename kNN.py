import numpy as np
import operator
import matplotlib.pyplot as plt
import os
def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels
def classify0(inX, dataSet, labels, k): #k-近邻算法
    dataSetSize = dataSet.shape[0]  #shape代表读取矩阵第一维度的长度
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet   #tile表示将输入的inX向量的行重复dataSetSize次，列重复1次组成矩阵
    sqDiffMat = diffMat**2  #**代表幂运算，相当于矩阵中每个数值的平方
    sqDistances = sqDiffMat.sum(axis=1) #axis=1表示按行累加，axis=0表示按列累加
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()    #argsort将数组从小到大进行排序，并且保存索引号即数组下标
    classCount={}
    for i in range(k):  #从0开始循环3个数字，根据索引号进行标记，第一个是B，则分类B的值+1，第二个是B，则分类B的值再+1，第三个是A，则分类A的值+1
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)   #sorted是进行排序操作，第一个元素是迭代对象，第二个是参照物（itemgetter表示序号），第三个True表示降序，False表示升序（默认）
    return sortedClassCount[0][0]
def file2matrix(filename):  #解析输入数据
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberofLines = len(arrayOLines)    #获取文件行数
    returnMat = np.zeros((numberofLines,3)) #zeros返回一个给定形状和类型的数组
    classlabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip() #strip()用于移除字符串头尾指定的字符（默认为空格）
        listFromLine = line.split('\t') #split(str="", num=string.count(str))通过指定分隔符对字符串进行切片，如果参数num 有指定值，则仅分隔 num 个子字符串，\t是水平制表（即TAB）
        returnMat[index,:] = listFromLine[0:3]
        classlabelVector.append(int(listFromLine[-1]))  #append()方法用于在列表末尾添加新的对象，索引值-1表示列表中最后一列元素
        index += 1
    return returnMat,classlabelVector
def autoNorm(dataSet):  #归一化特征值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m,1))
    normDataSet = normDataSet/np.tile(ranges, (m,1))
    return normDataSet, ranges, minVals
def datingClassTest():
    hoRatio = 0.10  #交叉训练比例，输入的文本数据中多少比例作为测试数据
    datingDataMat, datingLabels  =file2matrix('datingTestSet2.txt') #第一步处理读入的训练数据
    normMat, ranges, minVals = autoNorm(datingDataMat)  #第二步归一化特征值，将值确定在0-1的范围之内
    m = normMat.shape[0]
    numTestVece = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVece):
        classifierResult = classify0(normMat[i,:],normMat[numTestVece:m,:],datingLabels[numTestVece:m],3)
        print("the classifierResult came back with: %d, the real answer is: %d"%(classifierResult,datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f"%(errorCount/float(numTestVece)))
def classifyperson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)
    print("You will probably like this person:",resultList[classifierResult - 1])
def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect
def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('trainingDigits') #listdir用于返回指定的文件夹包含的文件或文件夹的名字的列表
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = os.listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))
select = int(input("请输入你要选择的操作："))
if select == 1:
    group, labels = createDataSet()
    print(classify0([0.5, 0.5], group, labels, 3))
elif select == 2:
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)  # add_subplot(xyz)的参数含义为划分x行，y列，将图片放在第z块
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * np.array(datingLabels), 15.0 * np.array(datingLabels))
    plt.show()
elif select == 3:
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
elif select == 4:
    datingClassTest()
elif select == 5:
    testVector = img2vector('testDigits/0_13.txt')
    print(testVector[0,0:31])
    print(testVector[0,32:63])
elif select == 6:
    classifyperson()
else:
    handwritingClassTest()