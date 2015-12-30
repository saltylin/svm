from numpy import *
from EnsembleSVM import *
from filereader import *
from LinearSVM import *
from extdim import *

def main():
    filename = ['../resource/train.csv','../resource/test.csv']
    itemid, numattr, cateattr, label = readfile(filename[0])
    testid, testnumattr, testcateattr = readfile(filename[1])
    multidim = MultiDimension(cateattr)
    trainextattr = multidim.gettrainextattr()
    testextattr = multidim.gettestextattr(testcateattr)
    trainattr = append(numattr, trainextattr, axis = 1)
    testattr = append(testnumattr, testextattr, axis = 1)
    model = EnsembleSVM(trainattr, label, LinearSVM)
    right = 0
    testnum = len(testattr)
    foutname = '../resource/result.csv'
    fout = open(foutname, 'w')
    fout.writeline('Id, Response')
    for i in range(testnum):
        p = model.predict(testattr[i])
        fout.writeline(testid[i] + ',' + p)
    fout.close()

if __name__ == '__main__':
    main()
