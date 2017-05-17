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
from keras.layers import Convolution2D, MaxPooling2D, Convolution3D, MaxPooling3D
from keras.layers import Merge, merge
from keras.utils import np_utils

from keras.models import model_from_json, Model
from keras import backend as K


import random as rdm
import nibabel as nib
import image_manager as imm
import gc

import tensorflow as tf
import json

#############################################

model_name	= "canadian_style"

#### default: takes 3 imag types -> less context so as the input size of 
 ### the NN is equal, and the comparison is fair


batch_size = 128
nb_classes = 2
nb_epoch = 350
# input image dimensions
inp_dim = 33
inp_dim_bigger = 65
step = 7

init_ler = 0.05
final_ler = 0.005
dec = (final_ler/init_ler)**(1/nb_epoch)

nb_rep = 1 # times you repeat each cross validation

nb_filters = 32
# size of pooling area for max pooling
pool_size = (4, 4)
# convolution kernel size
kernel_size = (3, 3)

#balance prop
bal_train = 10
bal_test = 200


# exp1
img_types = ["flair","FA","anatomica"]

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
#local_dropout1 = Dropout(0.25)(local_maxpool1)
print("Output shape of max pooling:", local_maxpool1.get_shape())

kernel_size = (3,3)
nb_filters = 64
pool_size = (2,2)
local_conv2 = Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape)(local_maxpool1)
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
final_model = model
#train_data partition
brains = ["tka002","tka003","tka004","tka005","tka006","tka007","tka009","tka010","tka011","tka012","tka013","tka015","tka016","tka017","tka018","tka019","tka020","tka021"]



tr = imm.ImageManager() # load training data
v = imm.ImageManager()
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


cv_history = []
for i in range(int(len(brains)/4)):
	
	for it in range(nb_rep):
		# train_brain = brains[:i*4]+brains[(i+1)*4:]
		# test_brain = brains[i*4:(i+1)*4]
		train_brain = ["tka002","tka003","tka004","tka005","tka006","tka009","tka011","tka012","tka013","tka016","tka017","tka019","tka020"]
		val_brain = ["tka010"]
		test_brain = ["tka007","tka015","tka018","tka021"]
		tr.reset()
		tr.init(train_brain)

		v.init(val_brain)
		v.createSlices(step=step+2)
		v.balance(bal_test)
		v.split(1)
		X_val_x = v.getData(img_types, "2dy", inp_dim)[0]
		y_val = X_val_x[1]
		X_val_x = X_val_x[0]
		X_val_bigger = v.getData(img_types, "2dy", inp_dim_bigger)[0][0]


		ler = init_ler
		for j in range(nb_epoch):
			print("Starting epoch:", j+1, "/", nb_epoch)
			print("Genrating new training data")
			# j = 0
			# while True:
			# 	try:
			# 		tr.prepareNewBatch(batch_size)
			# 	except Exception as e:
			# 		print(e)
			# 		break
			tr.createSlices(step=step)
			tr.balance(bal_train)
			tr.split(1)
			X_train_x = tr.getData(img_types, "2dy", inp_dim)[0]
			y_train = X_train_x[1]
			X_train_x = X_train_x[0]
			X_train_bigger =tr.getData(img_types, "2dy", inp_dim_bigger)[0][0]

			sgd = SGD(lr=ler,decay=0,momentum=0.0,nesterov = False)
			final_model.compile(loss='binary_crossentropy',
		              			optimizer=sgd,
		              				metrics=['accuracy'])
			tr_h = final_model.fit([X_train_bigger, X_train_x],y_train, batch_size=batch_size, nb_epoch=1,verbose=2)
			
			if (j+1)%1 == 0:
				val_loss = final_model.evaluate([X_val_bigger, X_val_x],y_val)
				score = evaluate(final_model,[X_val_bigger, X_val_x],y_val)
				tpr_val = score[1]
				tnr_val = score[2]
				cv_history.append((tr_h.history["loss"][0],val_loss[0],tnr_val,tpr_val))
			else:
				cv_history.append((tr_h.history["loss"][0],None,None,None))

			print(["train_loss", "val_loss", "tnr_val","tpr_val"])
			print(cv_history[-1])
			ler *= dec
			final_model.save("../models/model_" + model_name +"_it_"+str(j) + ".mdl")
			with open("hist_"+model_name+"_"+str(i)+".json","w") as tf:
				tf.write(json.dumps(cv_history))
				
		# train finished, 
		final_model.save("../models/model_" + model_name +"_"+ str(i) + ".mdl")

		with open("hist_"+model_name+"_"+str(i)+".json","w") as tf:
			tf.write(json.dumps(cv_history))

		#model = load_model("../models/model_0.mdl")
		tr.reset()
		### test stuff

		tt = imm.ImageManager() # load training data
		tt.init(test_brain)
		tt.createSlices(step=step+1)
		tt.balance(bal_test)
		tt.split(1) # we will select the hole brain
		X_test_x = tt.getData(img_types, "2dy", inp_dim)[0]
		y_test = X_test_x[1]
		X_test_x = X_test_x[0]
		X_test_bigger =tt.getData(img_types, "2dy", inp_dim_bigger)[0][0]

		score = evaluate(final_model,[X_test_bigger, X_test_x],y_test)
		res.append((score,train_brain, test_brain))
		print("###########################################")
		print("Cross Validation:",i, "\tIteration:",it)
		print("balance:", bal_test, "(", y_test.shape[0], ")")
		print(score[0][0])
		print(score[0][1])
		print("TPR:", score[1])
		print("TNR:", score[2])
		tt.reset()
	print("")
	print("########################################")
	print("###     Total (Average) cv: "+str(i)+"    ###")
	print("########################################")
	TP = 0
	TN = 0
	FP = 0
	FN = 0

	for el in res[i*nb_rep:(i*nb_rep)+nb_rep]:
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
	break

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



