import math
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
select = int(input("请输入你要选择的操作："))
if select == 1:
    myDat, labels = createDataSet()