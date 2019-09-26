import numpy as np
import cv2
import cv2.ml as ml
import sklearn.metrics as sk


from MachineLearn.Classes import *

KERNEL = "RBF"
MOLD = 8

oExp = Experiment()

oDataSet = DataSet()
base = np.loadtxt("GLCM/EXP_02/FEATURES_M1_CM8b.txt", usecols=range(24), delimiter=",")
classes = np.loadtxt("GLCM/EXP_02/FEATURES_M1_CM8b.txt", dtype=object, usecols=24, delimiter=",")
print(len(classes[classes == 'Class 1']))
print(len(classes[classes == 'Class 2']))
print(len(classes[classes == 'Class 3']))
print(len(classes[classes == 'Class 4']))
print(len(classes[classes == 'Class 5']))
print(len(classes[classes == 'Class 6']))
print(len(classes[classes == 'Class 7']))

for x, y in enumerate(base):
    oDataSet.add_sample_of_attribute(np.array(list(np.float32(y)) + [classes[x]]))
oDataSet.attributes = oDataSet.attributes.astype(float)
oDataSet.normalize_data_set()
for j in range(50):
    print(j)
    oData = Data(7, 11, samples=47)
    oData.random_training_test_by_percent([352, 382, 378, 382, 376, 360, 361], 0.8)
    svm = ml.SVM_create()
    svm.setKernel(ml.SVM_RBF)
    oData.params = dict(kernel=ml.SVM_RBF, kFold=2)
    svm.trainAuto(np.float32(oDataSet.attributes[oData.Training_indexes]), ml.ROW_SAMPLE,
                  np.int32(oDataSet.labels[oData.Training_indexes]), kFold=5)
    # svm.train_auto(np.float32(oDataSet.attributes[oData.Training_indexes]),
    #                np.float32(oDataSet.labels[oData.Training_indexes]), None, None, params=oData.params)
    results = []  # svm.predict_all(np.float32(oDataSet.attributes[oData.Testing_indexes]))
    for i in (oDataSet.attributes[oData.Testing_indexes]):
        res, cls = svm.predict(np.float32([i]))
        results.append(cls[0])
    oData.set_results_from_classifier(results, oDataSet.labels[oData.Testing_indexes])
    print(sk.accuracy_score(oDataSet.labels[oData.Testing_indexes].T[0],np.array(results).T[0]))
    oData.insert_model(svm)
    oDataSet.append(oData)
oExp.add_data_set(oDataSet,
                  description="  50 execucoes SVM_{} base FLUXO WON 24Att arquivos em FEATURES_M1_CM8b.txt. ".format(
                      KERNEL))
oExp.save("Objects/EXP02_SVM_{}_{}b.gzip".format(KERNEL, MOLD))

oExp = oExp.load("Objects/EXP02_SVM_{}_{}b.gzip".format(KERNEL, MOLD))

print(oExp)
print(oExp.experimentResults[0].sum_confusion_matrix / 50)
