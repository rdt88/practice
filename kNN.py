from numpy import *
import operator


def createdataset():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify(inX, dataset, labels, k):
    datasetsize = dataset.shape[0]
    diffmat = tile(inX, (datasetsize,1))-dataset
    sqdiffmat = diffmat**2
    sqdistances = sqdiffmat.sum(axis=1)
    distances = sqdistances**0.5
    sorteddistindicies = distances.argsort()
    classcount={}
    for i in range(k):
        voteilabel = labels[sorteddistindicies[i]]
        classcount[voteilabel] = classcount.get(voteilabel,0) + 1
    sortedclasscount = sorted(classcount.iteritems(),
        key=operator.itemgetter(1), reverse=True)
    return sortedclasscount[0][0]

#run with kNN.classify([0,0], group, labels, 3) in command line
#change [0,0] to see differnt guesses
#set up with import kNN and group, labels = kNN.createdataset()
