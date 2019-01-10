import numpy as np

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]    
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList,  classVec
                   
def createVocabList(dataSet):
    vacabSet = set([])
    for document in dataSet:
        vacabSet = vacabSet | set(document)
    return list(vacabSet)
    pass

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] *len(vocabList)
    #print(len(vocabList))
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word:%s is not in my vocabulary'%word)
    return returnVec
    pass

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] *len(vocabList)    
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print('the word:%s is not in my vocabulary'%word)
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
#    print('========================')
#    print(trainCategory)
#    print('========================')
#    print(numTrainDocs, numWords)
    pAbusive = sum(trainCategory) / float(numTrainDocs)
#    print(pAbusive)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if(trainCategory[i] == 1):
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive
    pass

def classifyNB(vec2Classify, p0Vec, p1Vec, pClasss1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClasss1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1 - pClasss1)
#    print('===========================')
#    print(p1Vec, len(p1Vec))
#    print('===========================')
#    print(vec2Classify, len(vec2Classify))
#    print('===========================')
#    print(vec2Classify * p1Vec)
#    print(p1, p0)
    if(p1 > p0):
        return 1
    else:
        return 0
    pass

def testingNB():
    listOPosts, listClasses = loadDataSet()    
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    #print(trainMat)
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
#    print(p0V)
#    print(p1V)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):        
        wordList = textParse(open('email/spam/%d.txt'%i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt'%i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if(classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]):
            errorCount += 1
    print('the error rate is: ', float(errorCount)/len(testSet))



'''['I', 'steak', 'dog', 'food', 'has', 'buying', 'help', 'problems', 'how', 'cute', 'licks', 
    'flea', 'stupid', 'take', 'park', 'so', 'garbage', 'mr', 'is', 'please', 'to', 'dalmation', 
    'my', 'maybe', 'not', 'quit', 'love', 'worthless', 'ate', 'stop', 'him', 'posting']'''

'''[[0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0], 
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1], 
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0], 
    [0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]]'''

if(__name__ == '__main__'):
    spamTest()
    #testingNB() 
    pass




