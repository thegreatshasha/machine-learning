#!/usr/bin/python

from math import log
import operator

# get the tree modules
import decision_tree
import tree_classify

def getDecisionTree(filename):
    fr = open(filename, 'r')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # lets see if the tree-file exists
    lensesTree = {}
    try:
        lensesTree = tree_classify.grabTree('lensesTree')
        print("Lenses tree file found.\n")    
    except IOError, AttributeError:
        print("Generating Lenses Tree.\n")
        # lens file not found
        lensesTree = decision_tree.createTree(lenses, lensesLabels)
        decision_tree.storeTree(lensesTree, 'lensesTree')
    # now we have our tree
    return lensesTree, lensesLabels
    
def prescribe(testVector):
    """
        Given a test-vector prescribe the type of contact lens that is suited
    """
    tree, labels = getDecisionTree('lenses.txt')
    ans = tree_classify.classify(tree, labels, testVector)
    print("Classifier output: %s" % ans)
    
if __name__ == '__main__':
    prescribe(['young', 'myope', 'yes', 'reduced']) # sample test vector
