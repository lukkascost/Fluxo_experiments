import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA
from numpy import linalg as LA

LABELS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24"]
attributes = np.loadtxt("../GLCM/EXP_01/FEATURES_M1_CM8b.txt", usecols=[x for x in range(0, 24)], delimiter=",")
index = np.array([x for x in range(len(attributes))])
labels = np.loadtxt("../GLCM/EXP_01/FEATURES_M1_CM8b.txt", usecols=-1, delimiter=",", dtype=object)


""" Unconditional analysis Mean, std and skewness results.
"""
resultTable = np.zeros((3, 24))
strres = ""
for i in range(24):
    resultTable[0, i] = np.mean(attributes[:, i])
    resultTable[1, i] = np.std(attributes[:, i])
    resultTable[2, i] = sp.skew(attributes[:, i])
    strres += "\\textit {}&{:.04f} & {:.04f} & {:.04f}\\\\\n".format(LABELS[i], resultTable[0, i],
                                                                             resultTable[1, i], resultTable[2, i])
print (strres)

""" class conditional analysis Mean, std and skewness results per class.
"""
resultTable = np.zeros((7, 3, 24))
for j in [1, 2, 3, 5, 6, 7]:
    for i in range(24):
        data = attributes[labels == "Class "+str(j), i]
        resultTable[j - 1, 0, i] = np.mean(data)
        resultTable[j - 1, 1, i] = np.std(data)
        resultTable[j - 1, 2, i] = sp.skew(data)

for i in range(24):
    strres = " \\textit {} ".format(LABELS[i])
    for j in [1, 2, 3, 5, 6, 7]:
        strres += "& {:.02f}$\pm${:.04f}".format(resultTable[j - 1, 0, i],resultTable[j - 1, 1, i])
    strres += "\\\\ "
    print (strres)
print
for i in range(24):
    strres = " \\textit{} ".format(LABELS[i])
    for j in [1, 2, 3, 5, 6, 7]:
        strres += "& {:.04f}".format(resultTable[j - 1, 2, i])
    strres += "\\\\ "
    print (strres)


""" Unconditional bi-variate analysis
"""
strres = ""
covTable = np.zeros((24, 24))
for i in range(24):
    strres += "\n\\textit{}".format(LABELS[i])
    for j in range(24):
        covTable[i, j] = np.corrcoef(attributes[:, i], attributes[:, j])[1, 0]
        strres += "&{:.04f}".format(covTable[i, j])
    strres += "\\\\"
print (strres)


""" Unconditional multi-variate analysis
"""
# mu = attributes.mean(axis=0)
# sigma = attributes.std(axis=0)
# attributes = (attributes - mu)/sigma
#
# resultPca = PCA(attributes, standardize=False)
# plt.clf()
# result = np.dot(attributes, resultPca.Wt)
# print (resultPca.fracs)
# for i in range(24):
#     plt.scatter(index, resultPca.Y[:, i], label="Attribute {}".format(i+1))
# plt.legend()
# plt.show()