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
from keras.layers import Convolution2D, MaxPooling2D
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
inp_dim = 13
step = 2
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

img_types = ["flair","anatomica", "FA"]

ex = sc.Examples()
ex.initilize()
ex.get_examples(step = step,output_type="classes")

ex.valance(10)

tot = ex.split(0.9)

X_train,y_train = tot[0]
X_train_x = ex.getData(X_train, img_types, "2dx", inp_dim)
X_train_y = ex.getData(X_train, img_types, "2dy", inp_dim)

X_test, y_test = tot[1]
X_test_x = ex.getData(X_test, img_types, "2dx", inp_dim)
X_test_y = ex.getData(X_test, img_types, "2dy", inp_dim)

X_train_x = np.asarray(X_train_x)
X_train_y = np.asarray(X_train_y)
y_train = np.asarray(y_train)
X_test_x = np.asarray(X_test_x)
X_test_y = np.asarray(X_test_y)
y_test = np.asarray(y_test)

print("len trains_x",len(X_train_x),len(y_train))
print("len trains_y",len(X_train_y),len(y_train))
print("len tests",len(X_test),len(y_test))





# print(ex.pairs[322])
# print("number of examples",len(ex.pairs))



# print("number of positive after selection: ",k)


# img_out = nib.load("../data/mask/normalized/tka003_lesion_mask_norm.nii.gz")
# data_out = imf.OurImage(img_out.get_data())

# k=0
# for x in data_out:
# 	if x >0:
# 		k+=1

# img_out = nib.load("../data/mask/normalized/tka004_lesion_mask_norm.nii.gz")
# data_out = imf.OurImage(img_out.get_data())


# for x in data_out:
# 	if x >0:
# 		k+=1

# print("number of positive before selection: ",k)


X_train_x = X_train_x.reshape(X_train_x.shape[0], inp_dim, inp_dim,len(img_types))
X_train_y = X_train_y.reshape(X_train_y.shape[0], inp_dim, inp_dim,len(img_types))
# unselect this for 3d images
#X_train = X_train.reshape(X_train.shape[0], inp_dim, inp_dim, inp_dim,1)


X_test_x = X_test_x.reshape(X_test_x.shape[0], inp_dim, inp_dim, len(img_types))
X_test_y = X_test_y.reshape(X_test_y.shape[0], inp_dim, inp_dim, len(img_types))
input_shape = (inp_dim, inp_dim, len(img_types))



# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255


# y_train2 = []
# y_test2 = []

# for i in range(len(y_train)):
# 	y_train2.append(y_train[i][1])

# for i in range(len(y_test)):
# 	y_test2.append(y_test[i][1])

# convert class vectors to binary class matrices (2 <-> binary classes)
# Y_train = np_utils.to_categorical(y_train2, 2)
# Y_test = np_utils.to_categorical(y_test2, 2)



# print(y_train[0])
# print(Y_train[0])

# print(y_train[1])
# print(Y_train[1])

# print(y_train[3])
# print(Y_train[3])
# quit()
# model = Sequential()


# model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
#                         border_mode='valid',
#                         input_shape=input_shape))
# model.add(Activation('relu'))

# model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
# model.add(Activation('relu'))

# model.add(MaxPooling2D(pool_size=pool_size))
# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(128))

# model.add(Activation('relu'))
# model.add(Dropout(0.5))

# model.add(Dense(nb_classes))
# model.add(Activation('softmax'))


model_x = Sequential()
print("Input shape to the network:", input_shape)
model_x.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model_x.add(Activation('relu'))
print("Output shape of 1st convolution:", model_x.output_shape)
model_x.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model_x.add(Activation('relu'))
print("Output shape of 2nd convolution:", model_x.output_shape)
model_x.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model_x.add(Activation('relu'))
print("Output shape of 3rd convolution:", model_x.output_shape)
model_x.add(MaxPooling2D(pool_size=pool_size))
model_x.add(Dropout(0.25))
print("Output shape after max pooling:", model_x.output_shape)



model_y = Sequential()
print("Input shape to the network:", input_shape)
model_y.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model_y.add(Activation('relu'))
print("Output shape of 1st convolution:", model_y.output_shape)
model_y.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model_y.add(Activation('relu'))
print("Output shape of 2nd convolution:", model_y.output_shape)
model_y.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model_y.add(Activation('relu'))
print("Output shape of 3rd convolution:", model_y.output_shape)
model_y.add(MaxPooling2D(pool_size=pool_size))
model_y.add(Dropout(0.25))
print("Output shape after max pooling:", model_y.output_shape)



merged = Merge([model_x, model_y], mode='concat')
final_model = Sequential()

final_model.add(merged)
print("Output shape after merge:", final_model.output_shape)
final_model.add(Flatten())
print("Output shape after flatten:", final_model.output_shape)
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

# json_string = final_model.to_json()
# f = open('../models/model.json', 'w')
# f.write(json_string)
# f.close()
# quit()


final_model.fit([X_train_x,X_train_y], y_train, batch_size=batch_size, validation_split=0.1, nb_epoch=nb_epoch,verbose=2)
final_model.save("../models/model_0.mdl")


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

score = evaluate(final_model,[X_test_x, X_test_y],y_test)
print(score[0][0])
print(score[0][1])
print("TPR:", score[1])
print("TNR:", score[2])


