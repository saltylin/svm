from numpy import *
from numpy.linalg import *

def LDA(attr, label):
    labeldict = {}
    for i in range(len(label)):
        tmp = label[i]
        if tmp not in labeldict:
            labeldict[tmp] = [i]
        else:
            labeldict[tmp].append(i)
    meanattrdict = {}
    attrnum = len(attr[0])
    totalmean = zeros((attrnum))
    for c in labeldict.keys():
        singlemeanattr = zeros((attrnum))
        for i in labeldict[c]:
            singlemeanattr += attr[i]
        totalmean += singlemeanattr
        singlemeanattr /= len(labeldict[c])
        meanattrdict[c] = singlemeanattr
    totalmean /= len(attr)
    Mw = zeros((attrnum, attrnum))
    for i in range(len(attr)):
        deviate = attr[i] - meanattrdict[label[i]]
        deviate.shape = (attrnum, 1)
        Mw += dot(deviate, transpose(deviate))
    #To make sure Mw is invertable
    diagsum = 0.0
    for i in range(len(Mw)):
        diagsum += Mw[i][i]
    diagsum /= len(Mw)
    diagsum *= 0.01
    Mw += diagsum * identity(len(Mw))
    Mb = zeros((attrnum, attrnum))
    for c in meanattrdict.keys():
        deviate = meanattrdict[c] - totalmean
        deviate.shape = (attrnum, 1)
        Mb += len(labeldict[c]) * dot(deviate, transpose(deviate))
    A = dot(inv(Mw), Mb)
    eigvalue, eigvector = eig(A)
    sortedarg = argsort(eigvalue.real)
    coe = []
    for i in range(len(eigvalue) - 1, len(eigvalue) - 8, -1):
        singlecoe = eigvector[:, sortedarg[i]].real
        coe.append(singlecoe)
    return coe

def conpress(attr, coe):
    result = zeros((len(attr), len(coe)))
    for i in range(len(attr)):
        for j in range(len(coe)):
            result[i][j] = dot(attr[i], coe[j])
    return result
