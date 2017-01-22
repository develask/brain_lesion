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

import tensorflow as tf
import json

#############################################

model_name	= "paralel_v1"

#### default: takes 3 imag types -> less context so as the input size of 
 ### the NN is equal, and the comparison is fair


batch_size = 128
nb_classes = 2
nb_epoch = 30
# input image dimensions
inp_dim_2d = 35
inp_dim_3d = 11
step = 8

# number of convolutional filters to use
nb_filters = 45
# size of pooling area for max pooling
pool_size_2d = (2, 2)
pool_size_3d = (2, 2, 2)
# convolution kernel size
kernel_size_2d = (3, 3)
kernel_size_3d = (3, 3, 3)


# exp1
img_types = ["flair","FA","anatomica"]



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
model_x.add(MaxPooling2D(pool_size=pool_size_2d))
model_x.add(Dropout(0.25))
print("Output shape after max pooling (2d):", model_x.output_shape)
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

model_y.add(MaxPooling2D(pool_size=pool_size_2d))
model_y.add(Dropout(0.25))

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

model_z.add(MaxPooling2D(pool_size=pool_size_2d))
model_z.add(Dropout(0.25))

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
model_3d.add(MaxPooling3D(pool_size=pool_size_3d))
model_3d.add(Dropout(0.25))
print("Output shape after max pooling (3d):", model_3d.output_shape)
model_3d.add(Flatten())
print("Output shape after flatten (3d):", model_3d.output_shape)


merged = Merge([model_x, model_y, model_z, model_3d], mode='concat')
final_model = Sequential()

final_model.add(merged)
print("Output shape after merge:", final_model.output_shape)
final_model.add(Dense(1024))
final_model.add(Activation('relu'))
final_model.add(Dropout(0.5))
print("Output shape after fully connected(dropout0.5):", final_model.output_shape)
final_model.add(Dense(128))
final_model.add(Activation('relu'))
final_model.add(Dropout(0.5))
print("Output shape after dully connected(dropout0.5):", final_model.output_shape)
final_model.add(Dense(nb_classes))
final_model.add(Activation('softmax'))
print("Output shape after softmax (2 classes):", final_model.output_shape)

quit()

#train_data partition
brains = ["tka002","tka003","tka004","tka005","tka006","tka007","tka009","tka010","tka011","tka012","tka013","tka015","tka016","tka017","tka018","tka019","tka020","tka021"]

#balance proportion
bal_train = 10
bal_test = 200

tr = imm.ImageManager() # load training data
res = []
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

	# mat[0][0] /= len(y_pred)
	# mat[0][1] /= len(y_pred)
	# mat[1][0] /= len(y_pred)
	# mat[1][1] /= len(y_pred)

	TPR = mat[0][0] / (mat[0][0] + mat[1][0])
	TNR = mat[1][1] / (mat[1][1] + mat[0][1])
	return(mat,TPR,TNR)	

for i in range(len(brains)/4):
	train_brain = brains[:i*4]+brains[(i+1)*4:]
	test_brain = brains[i*4:(i+1)*4]

	## load training data
	tr.reset()
	tr.init(train_brain)
	tr.createSlices(step=step)
	tr.balance(bal_train)
	tr.split(1) # we will select the hole brain

	X_train_x = tr.getData(img_types, "2dx", inp_dim_2d)[0]
	y_train = X_train_x[1]
	X_train_x = X_train_x[0]

	X_train_y = tr.getData(img_types, "2dy", inp_dim_2d)[0][0]

	X_train_z = tr.getData(img_types, "2dz", inp_dim_2d)[0][0]

	X_train_3d = tr.getData(img_types, "3d", inp_dim_3d)[0][0]




	final_model.compile(loss='binary_crossentropy',
	              optimizer='adadelta',
	              metrics=['accuracy'])

	cv = final_model.fit([X_train_x,X_train_y, X_train_z, X_train_3d], y_train, batch_size=batch_size, validation_split=0.1, nb_epoch=nb_epoch,verbose=2)
	final_model.save("../models/model_" + model_name +"_"+ str(i) + ".mdl")

	with open("hist_"+model_name+"_"+str(i)+".json","w") as tf:
		tf.write(json.dumps(cv.history))

	#model = load_model("../models/model_0.mdl")
	tr.reset()
	### test stuff

	tt = imm.ImageManager() # load training data
	tt.init(test_brain)
	tt.createSlices(step=step+1)
	tt.balance(bal_test)
	tt.split(1) # we will select the hole brain

	X_test_x = tt.getData(img_types, "2dx", inp_dim_2d)[0]
	y_test = X_test_x[1]
	X_test_x = X_test_x[0]

	X_test_y = tt.getData(img_types, "2dy", inp_dim_2d)[0][0]

	X_test_z = tt.getData(img_types, "2dz", inp_dim_2d)[0][0]

	X_test_3d = tt.getData(img_types, "3d", inp_dim_3d)[0][0]	

	score = evaluate(final_model,[X_test_x, X_test_y, X_test_z, X_test_3d],y_test)
	res.append((score,train_brain, test_brain))
	print("###########################################")
	print("Iteration:",i)
	print("balance:", bal_test, "(", y_test.shape[0], ")")
	print(score[0][0])
	print(score[0][1])
	print("TPR:", score[1])
	print("TNR:", score[2])
	tt.reset()

print("")
print("########################################")
print("###         Total (Average)          ###")
print("########################################")
TP = 0
TN = 0
FP = 0
FN = 0
for el in res:
	TP += el[0][0][0][0]
	TN += el[0][0][1][1]
	FP += el[0][0][0][1]
	FN += el[0][0][1][0]
total = TP+TN+FP+FN
TP = TP / float(total)
TN = TN / float(total)
FP = FP / float(total)
FN = FN / float(total)
print([TP,FP], ["TP","FP"])
print([FN,TN], ["FN","TN"])
print("")
print("Accuracy:", TP+TN)
print("TPR:", TP / float(TP + FN))
print("TNR:", TN / float(TN + FP))



