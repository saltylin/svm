from numpy import *

def readfile(filename):
    f = open(filename)
    attrname = f.readline().strip().split(',')
    attrname = attrname[1:]
    #determine the types of all attributes, i.e., numeric (0), categorical (1) and response (2)
    attrtype = analyzetype(attrname)
    numattrnum = 0
    cateattrnum = 0
    for t in attrtype:
        if t == 0:
            numattrnum += 1
        elif t == 1:
            cateattrnum += 1
    ###############3
    istrainingfile = True if attrtype[-1] == 2 else False
    data = f.readlines()
    f.close()
    numattr = zeros((len(data), numattrnum))
    cateattr = []
    itemid = []
    if istrainingfile:
        label = []
    index = 0
    for line in data:
        attrvalues = line.strip().split(',')
        itemid.append(int(attrvalues[0]))
        i, j = 0, 0
        k = 0
        singlecateattr = []
        while i < numattrnum or j < cateattrnum:
            val = attrvalues[k + 1]
            if attrtype[k] == 0:
                numattr[index][i] = float(val) if val != '' else 0.0
                i += 1
            else:
                singlecateattr.append(val)
                j += 1
            k += 1
        cateattr.append(singlecateattr)
        if istrainingfile:
            label.append(int(attrvalues[-1]))
        index += 1
    if istrainingfile:
        return itemid, numattr, cateattr, label
    else:
        return itemid, numattr, cateattr

def analyzetype(attrname):
    numattrstr = 'Product_Info_4, Ins_Age, Ht, Wt, BMI, Employment_Info_1, Employment_Info_4, Employment_Info_6, Insurance_History_5, Family_Hist_2, Family_Hist_3, Family_Hist_4, Family_Hist_5, Medical_History_1, Medical_History_10, Medical_History_15, Medical_History_24, Medical_History_32'
    numattrnameset = set(numattrstr.split(', '))
    attrtype = []
    for t in attrname:
        name = t[1:-1]
        if name in numattrnameset:
            attrtype.append(0)
        elif name == 'Response':
            attrtype.append(2)
        else:
            attrtype.append(1)
    return attrtype
