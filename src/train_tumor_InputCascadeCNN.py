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
from keras.layers import Dense, Dropout, Activation, Flatten, Input, InputLayer
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Merge, merge
from keras.utils import np_utils

from keras.models import model_from_json, Model
from keras import backend as K

import sample_cretation as sc
import random as rdm
import nibabel as nib
import image_functions as imf
import gc

import tensorflow as tf
    

#############################################

####### LOAD DATA ##########

batch_size = 128
nb_classes = 2
nb_epoch = 12

# input image dimensions
inp_dim = 33
inp_dim_bigger = 65

step = 3
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (4, 4)
# convolution kernel size
kernel_size = (3, 3)

#balance prop
bal_train = 10
bal_test = 10

img_types = ["flair","anatomica", "FA"]

train_brain = ["tka003","tka004"]
test_brain = ["tka002"]

ex = sc.Examples()
ex.initilize(crbs=train_brain)
ex.get_examples(step = step,output_type="classes")

ex.balance(bal_train)

tot = ex.split(1)

X_train,y_train = tot[0]
X_train_x = np.asarray(ex.getData(X_train, img_types, "2dy", inp_dim,crbs=train_brain))
X_train_bigger = np.asarray(ex.getData(X_train, img_types, "2dy", inp_dim_bigger,crbs=train_brain))


ex = sc.Examples()
ex.initilize(crbs=test_brain)
ex.get_examples(step = step,output_type="classes")

ex.balance(bal_test)

tot = ex.split(1)



X_test, y_test = tot[0]
X_test_x = np.asarray(ex.getData(X_test, img_types, "2dy", inp_dim,crbs=test_brain))
X_test_bigger = np.asarray(ex.getData(X_test, img_types, "2dy", inp_dim_bigger,crbs=test_brain))



X_train_x = X_train_x.reshape(X_train_x.shape[0], inp_dim, inp_dim,len(img_types))
X_train_bigger = X_train_bigger.reshape(X_train_bigger.shape[0], inp_dim_bigger, inp_dim_bigger,len(img_types))

X_test_x = X_test_x.reshape(X_test_x.shape[0], inp_dim, inp_dim, len(img_types))
X_test_bigger = X_test_bigger.reshape(X_test_bigger.shape[0], inp_dim_bigger, inp_dim_bigger, len(img_types))

input_shape = (inp_dim, inp_dim, len(img_types))

# Bigger path
input_shape_bigger = (inp_dim_bigger, inp_dim_bigger, len(img_types))

# Bigger path
big_window_input = Input(shape=input_shape_bigger,name="big_windows")
nb_filters = 5
kernel_size = (33,33)
print("Input shape to bigger layer:", input_shape_bigger)
big_window_conv1 = Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid')(big_window_input)
print("Output shape of 1st convolution:", big_window_conv1.get_shape())
big_window_relu1 = Activation('relu')(big_window_conv1)

# Small path
small_window_input = Input(shape=input_shape,name="small_windows")

merged_inputs = merge([big_window_relu1, small_window_input], mode='concat')

# Local path
kernel_size = (7,7)
nb_filters = 64
print("Input shape to the network:", merged_inputs.get_shape())
local_conv1 = Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid')(merged_inputs)
print("Output shape of 1st convolution:", local_conv1.get_shape())
local_relu1 = Activation('relu')(local_conv1)
print("Output shape of relu:", local_relu1.get_shape())
local_maxpool1 = MaxPooling2D(pool_size=pool_size,strides=(1,1))(local_relu1)
local_dropout1 = Dropout(0.25)(local_maxpool1)
print("Output shape of max pooling:", local_dropout1.get_shape())

kernel_size = (3,3)
nb_filters = 64
pool_size = (2,2)
local_conv2 = Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape)(local_dropout1)
print("Output shape of 2nd convolution:", local_conv2.get_shape())
local_relu2 = Activation('relu')(local_conv2)
print("Output shape of 2nd relu:", local_relu2.get_shape())
local_maxpool2 = MaxPooling2D(pool_size=pool_size,strides=(1,1))(local_relu2)
local_output = local_maxpool2

# Global path
model_y = Sequential()
nb_filters = 160
kernel_size = (13,13)
print("Input shape to the network:", merged_inputs.get_shape())
global_conv1 = Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape)(merged_inputs)
global_relu1 = Activation('relu')(global_conv1)
#global_dropout1 = Dropout(0.25)(global_relu1)
#print("Output dropout:", global_dropout1.get_shape())
global_output = global_relu1
# Merge paths TODO aquiiii cambiado Merge por merge
merged_paths = merge([local_output, global_output], mode='concat')
print("Output shape after merge:", merged_paths.get_shape())
nb_filters = nb_classes
kernel_size = (21,21)
merged_conv1 = Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid')(merged_paths)
merged_relu1 = Activation('relu')(merged_conv1)
print("Output shape of last convolution:", merged_relu1.get_shape())
merged_flatten1 = Flatten()(merged_relu1)
merged_dense1 = Dense(nb_classes)(merged_flatten1)
merged_softmax1 = Activation('softmax',name="output")(merged_dense1)
output = merged_softmax1
print("Output shape after softmax (2 classes):", merged_softmax1.get_shape())

model = Model(input=[big_window_input, small_window_input], output=output)

model.compile(loss='binary_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

print("gonna train")

# json_string = final_model.to_json()
# f = open('../models/model.json', 'w')
# f.write(json_string)
# f.close()
# quit()

model.fit([X_train_bigger,X_train_x], y_train, batch_size=batch_size, validation_split=0.1, nb_epoch=nb_epoch,verbose=2)
model.save("../models/model_0.mdl")


# model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#           verbose=1, validation_data=(X_test, Y_test))




#model = load_model("../models/model_0.mdl")
# score = model.evaluate(X_test, y_test, verbose=1)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])

def evaluate(model,X_test,y_test):
	y_pred = model.predict(X_test)
	mat = [[0,0],[0,0]] # [[TP,FP],[FN,TN]]
	for i in range(len(y_pred)):
		if y_test[i][1] == 0: # real negative
			mat[1][1] += y_pred[i][0] #TN
			mat[0][1] += y_pred[i][1] #FP
		else:
			mat[1][0] += y_pred[i][0] #FN
			mat[0][0] += y_pred[i][1] #TP

	mat[0][0] /= len(y_pred)
	mat[0][1] /= len(y_pred)
	mat[1][0] /= len(y_pred)
	mat[1][1] /= len(y_pred)

	TPR = mat[0][0] / (mat[0][0] + mat[1][0])
	TNR = mat[1][1] / (mat[1][1] + mat[0][1])
	return(mat,TPR,TNR)		

score = evaluate(model,[X_test_bigger, X_test_x],y_test)
print(score[0][0])
print(score[0][1])
print("TPR:", score[1])
print("TNR:", score[2])


