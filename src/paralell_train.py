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
inp_dim_2d = 19
inp_dim_3d = 9
step = 2

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size_2d = (2, 2)
pool_size_3d = (2, 2, 2)
# convolution kernel size
kernel_size_2d = (3, 3)
kernel_size_3d = (3, 3, 3)

# img types that will be taken as input
img_types = ["flair", "FA", "anatomica"]

#train_data partition
train_brain = ["tka003","tka004"]
test_brain = ["tka002"]

#balance proportion
bal_train = 10
bal_test = 10

## load training data

ex = sc.Examples()
ex.initilize(crbs=train_brain)
ex.get_examples(step = step,output_type="classes")
ex.balance(bal_train)
tot = ex.split(1)

X_train,y_train = tot[0]
X_train_x = np.asarray(ex.getData(X_train, img_types, "2dx", inp_dim_2d, crbs=train_brain))
X_train_y = np.asarray(ex.getData(X_train, img_types, "2dy", inp_dim_2d, crbs=train_brain))
X_train_z = np.asarray(ex.getData(X_train, img_types, "2dz", inp_dim_2d, crbs=train_brain))
X_train_3d = np.asarray(ex.getData(X_train, img_types, "3d", inp_dim_3d, crbs=train_brain))

print("size y_train",len(y_train))
## load test data

ex = sc.Examples()
ex.initilize(crbs=test_brain)
ex.get_examples(step = step,output_type="classes")
ex.balance(bal_test)
tot = ex.split(1)

X_test, y_test = tot[0]
X_test_x = np.asarray(ex.getData(X_test, img_types, "2dx", inp_dim_2d,crbs=test_brain))
X_test_y = np.asarray(ex.getData(X_test, img_types, "2dy", inp_dim_2d,crbs=test_brain))
X_test_z = np.asarray(ex.getData(X_test, img_types, "2dz", inp_dim_2d,crbs=test_brain))
X_test_3d = np.asarray(ex.getData(X_test, img_types, "3d", inp_dim_3d,crbs=test_brain))


# reshape it so its format is the requierd

X_train_x = X_train_x.reshape(X_train_x.shape[0], inp_dim_2d, inp_dim_2d,len(img_types))
X_train_y = X_train_y.reshape(X_train_y.shape[0], inp_dim_2d, inp_dim_2d,len(img_types))
X_train_z = X_train_y.reshape(X_train_z.shape[0], inp_dim_2d, inp_dim_2d,len(img_types))
X_train_3d = X_train_3d.reshape(X_train_3d.shape[0], inp_dim_3d, inp_dim_3d, inp_dim_3d,len(img_types))


X_test_x = X_test_x.reshape(X_test_x.shape[0], inp_dim_2d, inp_dim_2d, len(img_types))
X_test_y = X_test_y.reshape(X_test_y.shape[0], inp_dim_2d, inp_dim_2d, len(img_types))
X_test_z = X_test_y.reshape(X_test_z.shape[0], inp_dim_2d, inp_dim_2d, len(img_types))
X_test_3d = X_test_3d.reshape(X_test_3d.shape[0], inp_dim_3d, inp_dim_3d, inp_dim_3d,len(img_types))


# prepare input for the 

input_shape_2d = (inp_dim_2d, inp_dim_2d, len(img_types))
input_shape_3d = (inp_dim_3d, inp_dim_3d, inp_dim_3d, len(img_types))


## paralel NN, x
model_x = Sequential()
print("Input shape to the 2d networks:", input_shape_2d)
model_x.add(Convolution2D(nb_filters, kernel_size_2d[0], kernel_size_2d[1],
                        border_mode='valid',
                        input_shape=input_shape_2d))
model_x.add(Activation('relu'))
print("Output shape of 1st convolution (2d):", model_x.output_shape)
model_x.add(Convolution2D(nb_filters, kernel_size_2d[0], kernel_size_2d[1]))
model_x.add(Activation('relu'))
print("Output shape of 2nd convolution (2d):", model_x.output_shape)
model_x.add(Convolution2D(nb_filters, kernel_size_2d[0], kernel_size_2d[1]))
model_x.add(Activation('relu'))
print("Output shape of 3rd convolution (2d):", model_x.output_shape)
model_x.add(MaxPooling2D(pool_size=pool_size_2d))
model_x.add(Dropout(0.25))
print("Output shape after max pooling (2d):", model_x.output_shape)
model_x.add(Flatten())
print("Output shape after flatten (2d):", model_x.output_shape)



## paralel NN, y
model_y = Sequential()

model_y.add(Convolution2D(nb_filters, kernel_size_2d[0], kernel_size_2d[1],
                        border_mode='valid',
                        input_shape=input_shape_2d))
model_y.add(Activation('relu'))

model_y.add(Convolution2D(nb_filters, kernel_size_2d[0], kernel_size_2d[1]))
model_y.add(Activation('relu'))

model_y.add(Convolution2D(nb_filters, kernel_size_2d[0], kernel_size_2d[1]))
model_y.add(Activation('relu'))

model_y.add(MaxPooling2D(pool_size=pool_size_2d))
model_y.add(Dropout(0.25))

model_y.add(Flatten())



## paralel NN, z
model_z = Sequential()

model_z.add(Convolution2D(nb_filters, kernel_size_2d[0], kernel_size_2d[1],
                        border_mode='valid',
                        input_shape=input_shape_2d))
model_z.add(Activation('relu'))

model_z.add(Convolution2D(nb_filters, kernel_size_2d[0], kernel_size_2d[1]))
model_z.add(Activation('relu'))

model_z.add(Convolution2D(nb_filters, kernel_size_2d[0], kernel_size_2d[1]))
model_z.add(Activation('relu'))

model_z.add(MaxPooling2D(pool_size=pool_size_2d))
model_z.add(Dropout(0.25))

model_z.add(Flatten())




## paralel NN, 3d
model_3d = Sequential()
print("Input shape to the 3d network:", input_shape_3d)
model_3d.add(Convolution3D(nb_filters, kernel_size_3d[0], kernel_size_3d[1], kernel_size_3d[2],
                        border_mode='valid',
                        input_shape=input_shape_3d))
model_3d.add(Activation('relu'))
print("Output shape of 1st convolution (3d):", model_3d.output_shape)
model_3d.add(Convolution3D(nb_filters,kernel_size_3d[0], kernel_size_3d[1], kernel_size_3d[2],
						 border_mode='valid'))
model_3d.add(Activation('relu'))
print("Output shape of 2nd convolution (3d):", model_3d.output_shape)
#model_3d.add(MaxPooling3D(pool_size=pool_size_3d))
#model_3d.add(Dropout(0.25))
#print("Output shape after max pooling (3d):", model_3d.output_shape)
model_3d.add(Flatten())
print("Output shape after flatten (3d):", model_3d.output_shape)


merged = Merge([model_x, model_y, model_z, model_3d], mode='concat')
final_model = Sequential()

final_model.add(merged)
print("Output shape after merge:", final_model.output_shape)
final_model.add(Dense(128))
final_model.add(Activation('relu'))
final_model.add(Dropout(0.5))
print("Output shape after dropout(0.5):", final_model.output_shape)
final_model.add(Dense(nb_classes))
final_model.add(Activation('softmax'))
print("Output shape after softmax (2 classes):", final_model.output_shape)




final_model.compile(loss='binary_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

print("gonna train")

cv = final_model.fit([X_train_x,X_train_y, X_train_z, X_train_3d], y_train, batch_size=batch_size, validation_split=0.1, nb_epoch=nb_epoch,verbose=2)
final_model.save("../models/model_paralel.mdl")





#model = load_model("../models/model_0.mdl")


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

score = evaluate(final_model,[X_test_x, X_test_y, X_test_z, X_test_3d],y_test)
print(score[0][0])
print(score[0][1])
print("TPR:", score[1])
print("TNR:", score[2])


