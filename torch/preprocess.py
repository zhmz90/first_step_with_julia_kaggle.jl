#!/usr/local/env python

import h5py
from skimage.io import imread
import numpy as np
import pandas as pd

def read_data(typeData, labelsInfo, imageSize, path):
  x = np.zeros((labelsInfo.shape[0], 3, 20, 20))
  for (index, idImage) in enumerate(labelsInfo["ID"]):
    nameFile = "{0}/{1}Resized/{2}.Bmp".format(path, typeData, idImage)
    img = imread(nameFile, as_grey=False)
    #print(img.shape)
    #print(len(img.ravel()))
    x[index, :] = np.reshape(img.ravel(), (-1, 20, 20))
  return x
def class_dict(classes):
  num2class = pd.unique(classes) # num2class
  num2class.sort()
  class2num = {}
  for num,c in enumerate(num2class):
    class2num[c] = num
  return class2num,num2class

imageSize = 400
data_path = "/home/guo/haplox/Github/first_step_with_julia_kaggle/data/data"

labelsInfoTrain = pd.read_csv("{0}/trainLabels.csv".format(data_path))
xTrain = read_data("train", labelsInfoTrain, imageSize, data_path)
class2num,num2class = class_dict(labelsInfoTrain["Class"])
yTrain = map(lambda x:class2num[x], labelsInfoTrain["Class"])
yTrain = np.array(yTrain)

labelsInfoTest = pd.read_csv("{0}/sampleSubmission.csv".format(data_path))
xTest = read_data("test", labelsInfoTest, imageSize, data_path)
IDTest = labelsInfoTest["ID"]

classes = labelsInfoTrain["Class"]
class_dict(classes)


with h5py.File("data.hdf5", "w") as f:
    f.create_dataset("XTr", data = xTrain)
    f.create_dataset("yTr", data = yTrain)
    f.create_dataset("XTe", data = xTest)
    f.create_dataset("IDTe", data = IDTest)

