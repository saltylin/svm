from numpy import *
from EnsembleSVM import *
from filereader import *
from LinearSVM import *
from extdim import *
from WeightedModel import *

def main():
    filename = '../resource/train.csv'
    itemid, numattr, cateattr, label = readfile(filename)
    totalnum = len(numattr)
    testnum = totalnum * 0.1
    testnum = int(testnum)
    trainnum = totalnum - testnum
    trainnumattr = numattr[0: trainnum]
    traincateattr = cateattr[0: trainnum]
    trainlabel = label[0: trainnum]
    testnumattr = numattr[trainnum:]
    testcateattr = cateattr[trainnum:]
    testlabel = label[trainnum:]
    multidim = MultiDimension(traincateattr)
    trainextattr = multidim.gettrainextattr()
    testextattr = multidim.gettestextattr(testcateattr)
    trainattr = append(trainnumattr, trainextattr, axis = 1)
    testattr = append(testnumattr, testextattr, axis = 1)
    model = WeightedModel(trainattr, trainlabel)
    right = 0
    for i in range(testnum):
        p = model.predict(testattr[i])
        if p == testlabel[i]:
            right += 1
    accuracy = float(right) / testnum
    print 'accuracy:', accuracy

if __name__ == '__main__':
    main()
