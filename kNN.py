import numpy as np
import operator
import matplotlib.pyplot as plt
def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels
def classify0(inX, dataSet, labels, k):
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
def file2matrix(filename):
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
group,labels = createDataSet()
print(classify0([0.5,0.5], group, labels, 3))
datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
fig  = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
plt.show()