from numpy import *
import operator
import matplotlib.pyplot as plt
import os

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels
    pass

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    #print(dataSet)
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    #print(diffMat)
    sqDiffMat = diffMat**2
    #print(sqDiffMat)
    sqDistances = sqDiffMat.sum(axis=1)
    #print(sqDistances)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    #print(sortedDistIndicies)
    classCount={}
    for i in range(k):
        voreIlabel = labels[sortedDistIndicies[i]]
        classCount[voreIlabel] = classCount.get(voreIlabel, 0) + 1
    #print(classCount)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]
    pass

def file2matrix(filename):
    with open(filename) as fr:
        arrayOLines = fr.readlines()
        numberOfLines = len(arrayOLines)
        returnMat = zeros((numberOfLines, 3))
        #print(returnMat.shape)
        classLabelVector = []
        index = 0
        for line in arrayOLines:
            line = line.strip()
            listFromLine = line.split('\t')
            returnMat[index, :] = listFromLine[0:3]
            classLabelVector.append(int(listFromLine[-1]))
            index += 1
    return returnMat, classLabelVector
    pass

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    #print(minVals)
    maxVals = dataSet.max(0)
    #print(maxVals)
    ranges = maxVals - minVals
    #print(ranges)
    normDataSet = zeros(shape(dataSet))
    #print(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.05
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 5)
        print('the classifier came back with: %d, the real answer is: %d'%(classifierResult, datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    print('the total error rate is: %f'%(errorCount/float(numTestVecs)))
    

def drawPic():
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0*array(datingLabels), 15.0*array(datingLabels))
    pass



######################################################
#digital recognize
######################################################
def img2vector(filename):
    returnVec = zeros((1, 1024))
    with open(filename) as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVec[0, 32*i+j] = int(lineStr[j])
    return returnVec
    pass

def handWritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('digits/trainingDigits')
    m = len(trainingFileList)
    print(m)
    trainingMat = zeros((m , 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('digits/trainingDigits/%s'%fileNameStr)
    testFileList = os.listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s'%fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print('the classifier came back with: %d, the real answer is: %d' %(classifierResult, classNumStr))
        if(classifierResult != classNumStr):
            errorCount += 1.0
    print('\nthe total number of errors is: %d'%errorCount)
    print('\nthe total error rate is: %f'%(errorCount/float(mTest)))
        


if __name__ == '__main__':
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([10, 10000, 0.5])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print(classifierResult)
    pass































