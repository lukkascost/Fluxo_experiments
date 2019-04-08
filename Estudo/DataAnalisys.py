import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA
import matplotlib.cm as cm

attributes = np.loadtxt("../GLCM/EXP_01/FEATURES_M1_CM8b.txt", usecols=[x for x in range(0, 24)], delimiter=",")
index = np.array([x for x in range(len(attributes))])
labels = np.loadtxt("../GLCM/EXP_01/FEATURES_M1_CM8b.txt", usecols=-1, delimiter=",", dtype=object)
MK = [".", "_"]
CLASSES = ["Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6", "Class 7"]
ATTRIBUTES = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
              "20", "21", "22", "23", "24"]


COLOR = cm.rainbow(np.linspace(0, 1, attributes.shape[1]))
#
# """ Unconditional analysis: generate histogram for each attribute
#     independent of they class.
# """
# for i in range(attributes.shape[1]):
#     plt.clf()
#     plt.hist(attributes[:, i], 100)
#     plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
#     plt.savefig("FIGURE/hist_att-{:02d}_cls-{}.png".format(i + 1, "ALL"), dpi=100, bbox_inches="tight")
#
# """ class conditional analysis: generate histogram for each attribute in each class.
# """
# for i in range(attributes.shape[1]):
#     for j in CLASSES:
#         plt.clf()
#         x = index[labels == j]
#         y = attributes[x - 1, i]
#         plt.hist(y, 100)
#         plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
#         plt.savefig("FIGURE/hist_att-{:02d}_cls-{}.png".format(i + 1, j), dpi=100, bbox_inches="tight")
#
""" Unconditional bi-variate analysis
"""
covTable = np.zeros((attributes.shape[1], attributes.shape[1]))
for i in range(attributes.shape[1]):
    for j in range(attributes.shape[1]):
        covTable[i, j] = np.corrcoef(attributes[:, i], attributes[:, j])[1, 0]
plt.matshow(covTable)
plt.xticks(range(attributes.shape[1]), ATTRIBUTES)
plt.yticks(range(attributes.shape[1]), ATTRIBUTES)
plt.savefig("FIGURE/Covariance_table_-1.png", dpi=100, pad_inches=100)
plt.clf()
covTable = np.flip(covTable, axis=0)
plt.matshow(covTable)
plt.xticks(range(attributes.shape[1]), ATTRIBUTES)
plt.yticks(range(attributes.shape[1]), np.flip(ATTRIBUTES))
plt.savefig("FIGURE/Covariance_table_1.png", dpi=100, pad_inches=100)



# plt.clf()
# ok = []
# for i in range(attributes.shape[1]):
#     for j in range(attributes.shape[1]):
#         if i != j and not ([i, j] in ok):
#             plt.clf()
#             for w, k in enumerate(CLASSES):
#                 plt.scatter(attributes[labels == k, i], attributes[labels == k, j], label="class {}".format(k),
#                             color=COLOR[w], marker=MK[0])
#             plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
#             plt.xlabel("Predictor {}".format(i + 1))
#             plt.ylabel("Predictor {}".format(j + 1))
#             plt.savefig("FIGURE/att-{}_vs_{}_.png".format(i + 1, j + 1), dpi=300, bbox_inches="tight")
#             ok.append([i, j])
#             ok.append([j, i])
#
# """ Unconditional multi-variate analysis
# """
# mu = attributes.mean(axis=0)
# sigma = attributes.std(axis=0)
# attributes = (attributes - mu) / sigma
# attributes = np.nan_to_num(attributes)
# resultPca = PCA(attributes, standardize=False)
# plt.clf()
# result = np.dot(attributes, resultPca.Wt)
# for k, j in enumerate(CLASSES):
#     plt.scatter(resultPca.Y[labels == j, 0], resultPca.Y[labels == j, 1], label="class: {}".format(j), marker=MK[0],
#                 color=COLOR[k])
# plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
# plt.xlabel("PCA 1")
# plt.ylabel("PCA 2")
# plt.savefig("FIGURE/scatter_PCA_.png", dpi=300, bbox_inches="tight")
# plt.clf()
# for i in range(2):
#     for k, j in enumerate(CLASSES):
#         plt.scatter(index[labels == j], attributes[labels == j, i], label="Pre.: {} class: {}".format(i + 1, j),
#                     marker=MK[i], color=COLOR[k])
# plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
# plt.savefig("FIGURE/scatter_PCA_before.png", dpi=300, bbox_inches="tight")
