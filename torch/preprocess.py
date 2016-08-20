#!/usr/local/env python

import h5py
from skimage.io import imread
import numpy as np
import pandas as pd

def read_data(typeData, labelsInfo, imageSize, path):
  x = np.zeros((labelsInfo.shape[0], 20, 20, 3))
  for (index, idImage) in enumerate(labelsInfo["ID"]):
    nameFile = "{0}/{1}Resized/{2}.Bmp".format(path, typeData, idImage)
    img = imread(nameFile, as_grey=False)
    #print(len(img.ravel()))
    x[index, :] = np.reshape(img.ravel(), (20, 20, -1))
  return x


imageSize = 400
data_path = "/home/guo/haplox/Github/first_step_with_julia_kaggle/data/data"

labelsInfoTrain = pd.read_csv("{0}/trainLabels.csv".format(data_path))
xTrain = read_data("train", labelsInfoTrain, imageSize, data_path)
yTrain = map(ord, labelsInfoTrain["Class"])
yTrain = np.array(yTrain)

labelsInfoTest = pd.read_csv("{0}/sampleSubmission.csv".format(data_path))
xTest = read_data("test", labelsInfoTest, imageSize, data_path)
IDTest = labelsInfoTest["ID"]

with h5py.File("data.hdf5", "w") as f:
    f.create_dataset("XTr", data = xTrain)
    f.create_dataset("yTr", data = yTrain)
    f.create_dataset("XTe", data = xTest)
    f.create_dataset("IDTe", data = IDTest)
