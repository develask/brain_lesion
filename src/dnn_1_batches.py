from __future__ import print_function

from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Dense, Activation

import numpy as np
from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

from keras.models import load_model


from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Convolution3D, MaxPooling3D
from keras.layers import Merge
from keras.utils import np_utils

from keras.models import model_from_json
from keras import backend as K


import random as rdm
import nibabel as nib
import image_manager as imm
import gc
import ModelFunctions as MF

import tensorflow as tf
import json

#############################################

model_name	= "galician_DNN_softmax_v1"

#### default: takes 3 imag types -> less context so as the input size of 
 ### the NN is equal, and the comparison is fair


batch_size = 128
nb_classes = 2
nb_epoch = 2 #250
# input image dimensions
inp_dim_2d = 35
inp_dim_3d = 11
step = 19 #9

init_ler = 0.05
final_ler = 0.005
dec = (final_ler/init_ler)**(1/nb_epoch)

nb_rep = 1 # times you repeat each cross validation

# number of convolutional filters to use
nb_filters = 45 ### lets change it manually, increasing it 
# size of pooling area for max pooling
pool_size_2d = (2, 2)
pool_size_3d = (2, 2, 2)
# convolution kernel size
kernel_size_2d = (3, 3)
kernel_size_3d = (3, 3, 3)

#balance proportion
bal_train = 30
bal_test = 200

# exp1
img_types = ["flair","FA","anatomica"]



# prepare input for the 
input_shape_2d = (inp_dim_2d, inp_dim_2d, len(img_types))
input_shape_3d = (inp_dim_3d, inp_dim_3d, inp_dim_3d, len(img_types))


## paralel NN, x
model_x = Sequential()
print("Input shape to the 2d networks:", input_shape_2d)
model_x.add(Convolution2D(nb_filters-5, kernel_size_2d[0], kernel_size_2d[1],
                        border_mode='valid',
                        input_shape=input_shape_2d))
model_x.add(Activation('relu'))
print("Output shape of 1st convolution (2d):", model_x.output_shape)
model_x.add(Convolution2D(nb_filters, kernel_size_2d[0], kernel_size_2d[1]))
model_x.add(Activation('relu'))
print("Output shape of 2nd convolution (2d):", model_x.output_shape)
model_x.add(MaxPooling2D(pool_size=pool_size_2d))
#model_x.add(Dropout(0.25))
print("Output shape after max pooling (2d):", model_x.output_shape)
model_x.add(Convolution2D(nb_filters+10, kernel_size_2d[0], kernel_size_2d[1]))
model_x.add(Activation('relu'))
print("Output shape of 3rd convolution (2d):", model_x.output_shape)
model_x.add(MaxPooling2D(pool_size=pool_size_2d))
#model_x.add(Dropout(0.25))
print("Output shape after max pooling (2d):", model_x.output_shape)
model_x.add(Flatten())
print("Output shape after flatten (2d):", model_x.output_shape)



## paralel NN, y
model_y = Sequential()


model_y.add(Convolution2D(nb_filters-5, kernel_size_2d[0], kernel_size_2d[1],
                        border_mode='valid',
                        input_shape=input_shape_2d))
model_y.add(Activation('relu'))

model_y.add(Convolution2D(nb_filters, kernel_size_2d[0], kernel_size_2d[1]))
model_y.add(Activation('relu'))

model_y.add(MaxPooling2D(pool_size=pool_size_2d))
#model_y.add(Dropout(0.25))

model_y.add(Convolution2D(nb_filters+10, kernel_size_2d[0], kernel_size_2d[1]))
model_y.add(Activation('relu'))

model_y.add(MaxPooling2D(pool_size=pool_size_2d))
#model_y.add(Dropout(0.25))

model_y.add(Flatten())


## paralel NN, z
model_z = Sequential()

model_z.add(Convolution2D(nb_filters-5, kernel_size_2d[0], kernel_size_2d[1],
                        border_mode='valid',
                        input_shape=input_shape_2d))
model_z.add(Activation('relu'))

model_z.add(Convolution2D(nb_filters, kernel_size_2d[0], kernel_size_2d[1]))
model_z.add(Activation('relu'))

model_z.add(MaxPooling2D(pool_size=pool_size_2d))
#model_z.add(Dropout(0.25))

model_z.add(Convolution2D(nb_filters+10, kernel_size_2d[0], kernel_size_2d[1]))
model_z.add(Activation('relu'))

model_z.add(MaxPooling2D(pool_size=pool_size_2d))
#model_z.add(Dropout(0.25))

model_z.add(Flatten())


## paralel NN, 3d
model_3d = Sequential()
print("Input shape to the 3d network:", input_shape_3d)
model_3d.add(Convolution3D(nb_filters-5, kernel_size_3d[0], kernel_size_3d[1], kernel_size_3d[2],
                        border_mode='valid',
                        input_shape=input_shape_3d))
model_3d.add(Activation('relu'))
print("Output shape of 1st convolution (3d):", model_3d.output_shape)
model_3d.add(Convolution3D(nb_filters+10,kernel_size_3d[0], kernel_size_3d[1], kernel_size_3d[2],
						 border_mode='valid'))
model_3d.add(Activation('relu'))
print("Output shape of 2nd convolution (3d):", model_3d.output_shape)
model_3d.add(MaxPooling3D(pool_size=pool_size_3d))
#model_3d.add(Dropout(0.25))
print("Output shape after max pooling (3d):", model_3d.output_shape)
model_3d.add(Flatten())
print("Output shape after flatten (3d):", model_3d.output_shape)


merged = Merge([model_x, model_y, model_z, model_3d], mode='concat')
final_model = Sequential()

final_model.add(merged)
# print("Output shape after merge:", final_model.output_shape)
# final_model.add(Dense(1024))
# final_model.add(Activation('relu'))
# final_model.add(Dropout(0.5))
print("Output shape after merge:", final_model.output_shape)
final_model.add(Dense(128))
final_model.add(Activation('relu'))
print("Output shape after dully connected:", final_model.output_shape)
final_model.add(Dense(nb_classes))
final_model.add(Activation('softmax'))
print("Output shape after softmax (2 classes):", final_model.output_shape)


def getData(tr):
	tr.createSlices(step=step)
	tr.balance(bal_train)
	tr.split(1)
	X_train_x = tr.getData(img_types, "2dx", inp_dim_2d)[0]
	y_train = X_train_x[1]
	X_train_x = X_train_x[0]
	X_train_y = tr.getData(img_types, "2dy", inp_dim_2d)[0][0]
	X_train_z = tr.getData(img_types, "2dz", inp_dim_2d)[0][0]
	X_train_3d = tr.getData(img_types, "3d", inp_dim_3d)[0][0]

	return(([X_train_x,X_train_y, X_train_z, X_train_3d], y_train))


m = MF.Model(model_name, final_model, getData)
m.leaveOneOut(nb_epoch, batch_size, init_ler, final_ler)

