from numpy import *

class SVM:
    def __init__(self, trainattr, label, comid, label1, label2):
        self.trainattr = trainattr
        self.label = label
        self.comid = comid
        self.label1 = label1
        self.label2 = label2
        self.labeldict = {label1: 1, label2: -1}
