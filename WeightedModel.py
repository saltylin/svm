from numpy import *
from WeightedEnsembleSVM import *

class WeightedModel:
    def __init__(self, trainattr, trainlabel):
        instancenum = len(trainattr)
        weight = ones(instancenum) / instancenum
        self.ensemblelist = []
        self.alist = []
        for i in range(10):
            newensemble = WeightedEnsembleSVM(trainattr, trainlabel, weight)
            errorrate = 0.0
            flaglist = []
            for j in range(instancenum):
                p = newensemble.predict(trainattr[j])
                if p == trainlabel[j]:
                    flaglist.append(True)
                else:
                    errorrate += weight[j]
                    flaglist.append(False)
            a = 0.5 * math.log((1 - errorrate) / errorrate) + math.log(7)
            print a
            self.ensemblelist.append(newensemble)
            self.alist.append(a)
            for j in range(instancenum):
                if flaglist[j]:
                    weight[j] *= math.exp(-a)
                else:
                    weight[j] *= math.exp(a)
            zsum = sum(weight)
            weight /= zsum

    def predict(self, testattr):
        vote = {}
        for i in range(len(self.alist)):
            p = self.ensemblelist[i].predict(testattr)
            vote[p] = vote.get(p, 0.0) + self.alist[i]
        maxvote = 0.0
        for t in vote:
            if vote[t] > maxvote:
                maxvote = vote[t]
                result = t
        return result
