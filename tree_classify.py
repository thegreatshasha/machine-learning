#!/usr/bin/python

def classify(inputTree, labels, testVector):
    """
        Recursive routine to classify given an input tree
    """
    # get the root attribute in the decision tree
    firstStr = inputTree.keys()[0]
    # get the dictionary corresponding to this tree
    secondDict = inputTree[firstStr]
    # index of this attribute in the dataset can be obtained from the labels (list of all attribute names)
    index = labels.index(firstStr)
    # check the attribute of the testvector
    for key in secondDict.keys():
        if testVector[index] == key:
            # follow this branch
            if type(secondDict[key]).__name__ == 'dict':
                # recurse
                classLabel = classify(secondDict, labels, testVector)
            else:
                classLabel = secondDict[key]    # this is a leaf node
    return classLabel

    
def grabTree(filename):
    """
        Open a serialised tree
    """
    import pickle
    f = open(filename)
    return pickle.load(f)
    
