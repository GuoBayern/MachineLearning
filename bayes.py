import numpy as np
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1是侮辱性文字，0是正常言论
    return postingList, classVec
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:    #添加每篇文章中出现的新词
        vocabSet = vocabSet | set(document)
    return list(vocabSet)
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)    #获取输入表的长度，建立一个所有元素都为0的向量
    for word in inputSet:   #遍历需要检查的集合，有匹配的将值变为1
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!"%word)
    return returnVec
select = int(input("请输入你要选择的操作："))
def trainNB0(trainMatrix, trainCategory):   #trainMatrix是经过01转换的矩阵
    numTrainDocs = len(trainMatrix) #获取训练矩阵的长度
    numWords = len(trainMatrix[0])  #获取训练矩阵的列数
    pAbusive = sum(trainCategory)/float(numTrainDocs)   #表示侮辱类概率
    p0Num = np.zeros(numWords)
    p1Num = np.zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num/p1Denom
    p0Vect = p0Num/p0Denom
    return p0Vect, p1Vect, pAbusive
if select == 1:
    listPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listPosts)
    print(setOfWords2Vec(myVocabList, listPosts[0]))
    print(setOfWords2Vec(myVocabList, listPosts[3]))
elif select == 2:
    listPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listPosts)
    print(setOfWords2Vec(myVocabList, listPosts[0]))
    print(setOfWords2Vec(myVocabList, listPosts))