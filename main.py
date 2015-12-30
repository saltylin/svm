from numpy import *
from EnsembleSVM import *
from filereader import *
from LinearSVM import *
from extdim import *

def main():
    filename = '../resource/train.csv'
    itemid, numattr, cateattr, label = readfile(filename)
    totalnum = len(numattr)
    testnum = totalnum / 10
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
    model = EnsembleSVM(trainattr, trainlabel, LinearSVM)
    right = 0
    for i in range(trainnum):
        p = model.predict(trainattr[i])
        if p == trainlabel[i]:
            right += 1
    accuracy = float(right) / trainnum
    print 'accuracy:', accuracy

if __name__ == '__main__':
    main()
