import math
import operator
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]  #数组索引值-1，表示读取这个数组的最后一个
        if currentLabel not in labelCounts.keys():  #判断最后的一个标签是否在分类中，如果在则分类统计+1，分类不在的话就加入分类
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * math.log(prob,2)   #计算香农熵
    return shannonEnt
def createDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels
def splitDataSet(dataSet, axis, value): #按照特征划分数据集后，每一次决策完成返回的数据将会去掉这一次决策的特征值
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis] #表示从0-axis，但是不取axis这个下标
            reducedFeatVec.extend(featVec[axis + 1:])   #从axis+1到最后一个值，都输出，extend()函数用于在列表末尾一次性追加另一个序列中的多个值
            retDataSet.append(reducedFeatVec)
    return retDataSet
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1   #获取类别标签数组索引值
    baseEntropy = calcShannonEnt(dataSet)   #计算数据集原始香农熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]  #将每一列数据循环遍历放入到新的列表中
        uniqueVals = set(featList)      #set()创建一个不重复的元素集合，可以取出每一列特征元素的唯一元素值
        newEntropy = 0.0
        for value in uniqueVals:    #根据每一列特征元素的特征值计算香农熵
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
def majorityCnt(classList): #与k-近邻算法中的分类器排名相类似
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): #类别标签完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1:    #遍历完所有特征元素，返回出现次数最多的类别标签
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree
select = int(input("请输入你要选择的操作："))
if select == 1:
    myDat, labels = createDataSet()
elif select == 2:
    myDat, labels = createDataSet()
    print(splitDataSet(myDat, 0 ,1))
elif select == 3:
    dataSet1 = [[1, 'yes'], [1, 'yes'], [0, 'no'], [1, 'no'], [1, 'no']]
    dataSet2 = [[1, 'yes'], [1, 'yes'], [1, 'no'], [0, 'no'], [0, 'no']]
    print(calcShannonEnt(dataSet1))
    print(calcShannonEnt(dataSet2))
elif select == 4:
    myDat, labels = createDataSet()
    print(chooseBestFeatureToSplit(myDat))