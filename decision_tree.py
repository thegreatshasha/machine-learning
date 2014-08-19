#!/usr/bin/python

# standard imports
from math import log
import operator

def calcShannonEntropy(dataSet):
    """
        Returns the Shannon entropy of the entire dataset
        Information of x => l(x) = log2p(x)
        Shannon entropy = - sum over all x {p(xi)*l(xi)}
    """
    numEntries = len(dataSet)
    # store all labels in a dictionary in the format (label: count)
    labelCounts = {}
    for featVect in dataSet:
        label = featVect[-1]
        if label not in labelCounts:
            labelCounts[label] = 0
        labelCounts[label] += 1
    shannonEnt = 0.0
    for key in labelCounts.keys():
        prob = float(labelCounts[key]) / numEntries
        # calculate information
        l = log(prob, 2)
        shannonEnt -= (prob * l)
    return shannonEnt   
    
def splitDataSet(dataSet, axis, value):
    """
        Helper function to split the dataset on an axis
    """
    # lists are passed by reference...we will call this on same list multiple times
    # we do not want the list to change
    retDataList = []
    for featVect in dataSet:
        if featVect[axis] == value:
            reducedFeatVect = featVect[:axis]   # axis is not included
            reducedFeatVect.extend(featVect[axis+1:])   # this consumes attributes
            retDataList.append(reducedFeatVect)
    return retDataList
    
def chooseBestFeatureToSplit(dataSet):
    """
        Returns the index of the best feature to split on
        It tries to maximise the information gain
        information_gain = baseEntropy - newEntropy
        as entropy is the degree of randomness
    """
    numFeatures = len(dataSet[0]) - 1   # as we expect last element to be the class
    # get the base entropy before splitting
    baseEntropy = calcShannonEntropy(dataSet)
    # keep track of best information gain
    bestInfoGain = 0.0
    bestFeature = -1    # not found yet
    for i in range(numFeatures):
        featValues = [vector[i] for vector in dataSet]
        uniqueVals = set(featValues)
        newEntropy = 0.0    # re-initialised to 0 for every feature
        for val in uniqueVals:
            # iterate over all values of feature
            # sum up the entropy and compare information gain
            subDataSet = splitDataSet(dataSet, i, val)
            prob = len(subDataSet) / float(len(dataSet))    # something has to be float else we lose precision
            newEntropy += prob * calcShannonEntropy(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCount(classList):
    """
        When we dont have any more attributes to look into
        and when all elements are not in the same class
        group by a majority vote
    """
    classCount = {}
    for elem in classList:
        if elem not in classCount.keys():
            classCount[elem] = 0
        classCount[elem] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
    

def createTree(dataSet, labels):
    """
        Recursive routine to create a decision tree
    """
    classList = [vector[-1] for vector in dataSet]  # as last element in class name
    if len(classList) == classList.count(classList[0]):
        # all elements are of same class...this is a leaf node
        return classList[0]
    if len(dataSet[0]) == 1:
        # no more features to split on...take a majority vote
        return majorityCount(classList)
    # base cases have been accounted for
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestLabel = labels[bestFeat]
    myTree = {bestLabel:{}}
    del(labels[bestFeat])
    featValues = [vector[bestFeat] for vector in dataSet]
    uniqueVals = set(featValues)
    for val in uniqueVals:
        subLabels = labels[:] # we do not want our labels tampered with as this algo goes depth-first
        myTree[bestLabel][val] = createTree(splitDataSet(dataSet, bestFeat, val), subLabels)
    return myTree
    


def storeTree(inputTree, filename):
    """
        Building the tree is enormous work. Serialise the dictionary so that it can be obtained quickly later
    """
    import pickle
    f = open(filename, 'w')
    pickle.dump(inputTree, f)
    f.close()
    
