from numpy import ndarray
from WeightedLinearSVM import *

class WeightedEnsembleSVM:
    def __init__(self, trainattr, label, weight):
        labeldict = {}
        for i in range(len(label)):
            tmplabel = label[i]
            if tmplabel in labeldict:
                labeldict[tmplabel].append(i)
            else:
                labeldict[tmplabel] = [i]
        self.labelnum = len(labeldict)
        self.svmlist = []
        for i in range(2, self.labelnum + 1):
            for j in range(1, i):
                comid = labeldict[i] + labeldict[j]
                tmpsvm = WeightedLinearSVM(trainattr, label, comid, i, j, weight)
                self.svmlist.append(tmpsvm)

    def predict(self, testattr):
        vote = {}
        for t in self.svmlist:
            p = t.predict(testattr)
            vote[p] = vote.get(p, 0) + 1
        maxvote = 0
        for t in vote:
            if vote[t] > maxvote:
                result = t
                maxvote = vote[t]
        return result
