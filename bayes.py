import numpy as np
import math
import re
import feedparser as fp
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
def setOfWords2Vec(vocabList, inputSet):    #朴素贝叶斯的词集模型
    returnVec = [0] * len(vocabList)    #获取输入表的长度，建立一个所有元素都为0的向量
    for word in inputSet:   #遍历需要检查的集合，有匹配的将值变为1
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!"%word)
    return returnVec
def trainNB0(trainMatrix, trainCategory):   #trainMatrix是经过01转换的矩阵
    numTrainDocs = len(trainMatrix) #获取训练矩阵的长度
    numWords = len(trainMatrix[0])  #获取训练矩阵的列数
    pAbusive = sum(trainCategory)/float(numTrainDocs)   #表示侮辱类概率
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num / p1Denom
    p1Vect = [math.log(x) for x in p1Vect]
    p0Vect = p0Num / p0Denom
    p0Vect = [math.log(x) for x in p0Vect]
    return p0Vect, p1Vect, pAbusive
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + math.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + math.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V ,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))
def bagOfWords2VecMN(vocabList, inputSet):  #朴素贝叶斯的词袋模型
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec
def textParse(bigString):
    listOfTokens = re.split(r'\W*', bigString)  #使用split函数根据除了数字和英文字母以外的字符做分割
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]    #转换为小写字母
def spamTest():
    docList=[]
    classList = []
    fullText =[]
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' %i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #在读取所有spam文件夹中的文件后，最后加入标记1
        wordList = textParse(open('email/ham/%d.txt' %i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0) #在读取所有ham文件夹中的文件后，最后加入标记0
    vocabList = createVocabList(docList)
    trainingSet = range(50)
    testSet=[]
    for i in range(10): #随机选择其中的10份邮件
        randIndex = int(np.random.uniform(0, len(trainingSet))) #uniform(x,y)表示在[x,y)之间随机生成一个实数
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error",docList[docIndex])
    print('the error rate is: ',float(errorCount) / len(testSet))
select = int(input("请输入你要选择的操作："))
if select == 1:
    listPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listPosts)
    print(setOfWords2Vec(myVocabList, listPosts[0]))
    print(setOfWords2Vec(myVocabList, listPosts[3]))
elif select == 2:
    listPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listPosts)
    trainMat = []
    for postinDoc in listPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    print(pAb)
    print(p0V)
    print(p1V)
elif select == 3:
    testingNB()
elif select == 4:
    emailText = open('email/ham/6.txt').read()
    regEx = re.compile('\\W*')
    print(regEx.split(emailText))
elif select == 5:
    spamTest()
elif select == 6:
    ny = fp.parse('http://newyork.craigslist.org/stp/index.rss')
    print(len(ny['entries']))
