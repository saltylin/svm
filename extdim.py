from numpy import *

class MultiDimension:
    def __init__(self, cateattr):
        instancenum = len(cateattr)
        self.attrnum = len(cateattr[0])
        self.valdictlist = []
        for i in range(self.attrnum):
            valdict = {}
            valindex = 0
            for j in range(instancenum):
                tmpval = cateattr[j][i]
                if tmpval not in valdict:
                    valdict[tmpval] = valindex
                    valindex += 1
            self.valdictlist.append(valdict)
        self.extendnum = 0
        for t in self.validictlist:
            self.extendnum += len(t)
        self.trainextattr = extend(self, cateattr)

    def gettrainextattr(self):
        return self.trainextattr

    def gettestextattr(self, cateattr):
        return extend(self, cateattr)

    def extend(self, cateattr):
        instancenum = len(cateattr)
        result = zeros((instancenum, self.extendnum))
        start = 0
        for i in range(self.attrnum):
            for j in range(instancenum):
                index = start + self.valdictlist[i].get(cateattr[j][i], -1)
                if index != -1:
                    result[j][index] = 1.0
            start += len(self.valdictlist[i])
        return result
