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

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    print(numTrainDocs, numWords)
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = np.zeros(numWords)
    p1Num = np.zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs):
        if(trainCategory[i] == 1):
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num/p1Denom
    p0Vect = p0Num/p0Denom
    return p0Vect,p1Vect,pAbusive
    pass

if(__name__ == '__main__'):
    listOPosts, listClasses = loadDataSet()
    #print(listOPosts.count())
    myVocabList = createVocabList(listOPosts)
#    print(myVocabList)
#    setOfWords2Vec(myVocabList, listOPosts[0])
#    print(listOPosts[0], type(listOPosts[0]))
#    print(setOfWords2Vec(myVocabList, listOPosts[1]))
#    print(setOfWords2Vec(myVocabList, ['your', 'cat', 'wrong']))
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    print(trainMat)
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    print(p0V, p1V, pAb)
    pass




