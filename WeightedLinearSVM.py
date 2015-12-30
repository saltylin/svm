from numpy import *
from SVM import *

class WeightedLinearSVM(SVM):
    def __init__(self, trainattr, label, comid, label1, label2, weight):
        SVM.__init__(self, trainattr, label, comid, label1, label2)
        self.weight = weight
        self.coe = 0.01
        self.T = int(1 * self.instancenum)
        self.w = zeros(self.attrnum)
        self.b = 0.0
        self.training()

    def training(self):
        for t in range(1, self.T + 1):
            randomindex = random.randint(self.instancenum)
            yt = 1.0 / (self.coe * t)
            trainitem = self.trainattr[self.comid[randomindex]]
            weight = self.weight[self.comid[randomindex]] * len(self.trainattr)
            innerproduct = dot(self.w, trainitem)
            tmplabel = self.labeldict[self.label[self.comid[randomindex]]]
            if innerproduct * tmplabel < 1:
                self.w = (1.0 - yt * self.coe) * self.w + yt * tmplabel * trainitem * weight
                self.b = (1.0 - yt * self.coe) * self.b + yt * tmplabel * weight
            else:
                self.w = (1.0 - yt * self.coe) * self.w
                self.b = (1.0 - yt * self.coe) * self.b

    def predict(self, testattr):
        flag = self.b
        flag += dot(self.w, testattr)
        result = self.label1 if flag >= 0 else self.label2
        return result
