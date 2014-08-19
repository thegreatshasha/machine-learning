#!/usr/bin/python

from numpy import *
import operator
from os import listdir

def img2vector(filename, x_res=32, y_res=32):
    """
        Convert 32x32 images (0s and 1s) into a 1x1024 array
    """
    f = open(filename, 'r')
    returnVector = zeros((1, 1024))
    for i in range(y_res):
        line = f.readline()
        for j in range(x_res):
            returnVector[0, 32*i + j] = int(line[j])
    return returnVector
    
def classify0(inX, dataSet, labels, k):
    """
        inX: input Vector to be classified
        dataSet: training data
        labels: training data classes
        k: k for k nearest neighbors algorithm
    """
    # get the dataset size
    dataSetSize = dataSet.shape[0]
    # get (x1-x2) (y1-y2) type matrix
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # (x1-x2)^2 (y1-y2)^2 type matrix
    sqDiffMat = diffMat**2
    # get the distance add the squared delta x and delta y
    sqDistances = sqDiffMat.sum(axis=1)
    # take a root to get the distance
    distances = sqDistances**0.5
    # get the indices that sort the distances in ascending order
    sortedDistIndices = distances.argsort()
    
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]
    
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('training_digits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = (int)(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('training_digits/%s' % fileNameStr)
    
    # testing
    testFileList = listdir('test_digits')
    errorCount = 0.0
    mTest = len(testFileList)
    
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = (int)(fileStr.split('_')[0])
        vectorUnderTest = img2vector('test_digits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("Classifier result: %d\tReal Answer: %d\n" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("\nTotal Number of Errors: %d\n" % errorCount)
    print("Total Error rate: %f\n" % (errorCount/float(mTest)))
    
if __name__ == '__main__':
    handwritingClassTest()
    
