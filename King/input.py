import string
import pandas as pd
import numpy as np
from numpy.random import shuffle
#import skimage.io import imread
from scipy.misc import imread
import tensorflow as tf

tf.app.flags.DEFINE_boolean("debug", True, "for debug models")
tf.app.flags.DEFINE_boolean("use_fp16", False, "data type")
FLAGS = tf.app.flags.FLAGS


data_dir = "/home/guo/haplox/Github/first_step_with_julia_kaggle/data/data"
image_size = 400
labels = string.digits + string.lowercase + string.uppercase
label_dict = {l:i for (i,l) in enumerate(labels)}
num_class = len(labels) # 62

def data_type():
    if FLAGS.use_fp16:
        return tf.float16
    else:
        return tf.float32

def read_data(type_data, labels_info, path):
    x = np.zeros((labels_info.shape[0], 20, 20, 3)) # 20 x 20 x 3 image
    for (index, id_image) in enumerate(labels_info["ID"]):
        file_name = "{0}/{1}Resized/{2}.Bmp".format(path, type_data, id_image)
        img = imread(file_name)
        try:
            x[index,:] = img
        except:
            print(file_name)
    return x


# Read train matrix
labels_info_train = pd.read_csv("{0}/trainLabels.csv".format(data_dir))
xTrain = read_data("train", labels_info_train, data_dir)

# Read test matrix
labels_info_test = pd.read_csv("{0}/sampleSubmission.csv".format(data_dir))
xTest  = read_data("test", labels_info_test, data_dir)

yTrain = map(lambda x:label_dict[x], labels_info_train["Class"])
print(yTrain[1:10])
yTrain = np.array(yTrain)
print(map(lambda x: labels[x], yTrain[1:10]))

print(labels_info_test.head())

### shuffle and split dataset into tr and val dataset
def shuffle_split_data(data, labels):
    np.random.seed(201607)
    shuffle(data)
    np.random.seed(201607)
    shuffle(labels)
    num_example = len(labels)
    num_tr = int(num_example * 0.8)
    return (data[0:num_tr],labels[0:num_tr],data[num_tr:],labels[num_tr:])

xTr,yTr,xVal,yVal = shuffle_split_data(xTrain,yTrain)

### train model with above data
def model():
    


