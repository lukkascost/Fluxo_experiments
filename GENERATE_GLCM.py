import cv2
import numpy as np

from MachineLearn.Classes.Extractors.GLCM import GLCM

MIN_BITS = 8
MAX_BITS = 8

MIN_DECIMATION = 4
MAX_DECIMATION = 4

PATH_TO_IMAGES_FOLDER = 'D:/Repository/Dataset-Fluxo/Won_data_m4/'
PATH_TO_SAVE_FEATURES = 'GLCM/EXP_02/'

for nbits in range(MIN_BITS, MAX_BITS + 1):
    for k in range(MIN_DECIMATION, MAX_DECIMATION + 1):
        listGLCM = []
        for quantity in [[1, 352], [2, 382], [3, 378], [4, 382], [5, 376], [6, 360], [7, 361]]:
            for image in range(1, quantity[1] + 1):
                img = cv2.imread(PATH_TO_IMAGES_FOLDER + "C{0}/c{0}_{1}.png".format(quantity[0], image),0)
                """ DECIMATION """
                # klist = [x for x in range(0, img.shape[0], k)]
                # klist2 = [x for x in range(0, img.shape[1], k)]
                # img = img[klist]
                # img = img[:, klist2]

                """ CHANGING IMAGE TO VALUES BETWEEN 0 AND  2**NBITS"""
                # img = img / 2 ** (12 - nbits)

                """ APPLYING OTSU'S ALGORITHM """
                # ret, img = cv2.threshold(img, 0, (2 ** nbits)-1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                """ GENERATING FEATURES FOR GLCM """
                oGlcm = GLCM(img, nbits)
                oGlcm.generateCoOccurenceHorizontal()
                oGlcm.normalizeCoOccurence()
                oGlcm.calculateAttributes()

                """ ADDING FEATURES IN ARRAY FOR SAVE IN FILE """
                listGLCM.append(oGlcm.exportToClassfier("Class " + str(quantity[0])))
                print(nbits, k, quantity[0], image)
                listGLCM2 = np.array(listGLCM)

                """ SAVE FILE WITH FEATURES, DECIMATION WITH STEP = k AND CORRELATION MATRIX WITH nbits BITS. """
                np.savetxt(PATH_TO_SAVE_FEATURES + "FEATURES_M{}_CM{}b.txt".format(k, nbits), listGLCM2, fmt="%s",
                           delimiter=',')
