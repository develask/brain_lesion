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
from keras.utils import np_utils
from keras import backend as K

import sample_cretation as sc
import random as rdm
import nibabel as nib
import image_functions as imf
import gc
    

#############################################

####### LOAD DATA ##########

batch_size = 128
nb_classes = 2
nb_epoch = 12

# input image dimensions
inp_dim = 5
step = 3

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

ex = sc.Examples()
ex.get_examples("flair","2dx",inp_dim,step = step,output_type="classes")

# import time
# print("cata la ram")
# time.sleep(5)

num_pos = 0
for pair in ex.pairs:
	if pair[1] == 1:
		num_pos += 1

r = list(range(0,len(ex.pairs)))
rdm.shuffle(r)

num_neg = 0
to_be_removed = [] # list of index that should be removed
for i in r:
	if ex.pairs[i][1] == 0:
		if num_neg < num_pos:
			num_neg += 1
		else:
			to_be_removed.append(i)

to_be_removed = sorted(to_be_removed,reverse=True)
print("toberemoved",len(to_be_removed))
for i in to_be_removed:
	ex.remove_elem(i)


print("len of pairs",len(ex.pairs))

ex.shuffle_exs() ## shuffle the training - test examples

train_test = 0.8 # train test proportion
i = 0
train = []
test = []
for pair in ex.pairs:
	if i<0.8*len(ex.pairs):
		train.append(pair)
		i += 1
	else:
		test.append(pair)

# print("voy a borrar pairs")
# time.sleep(5)

# ex.reset_exs() # liberate the RAM a little,
# del ex
# gc.collect()

# print("he borrado")
# time.sleep(5)

X_train = []
y_train = []


for i in range(len(train)):
	X_train.append(train[i][0])
	y_train.append(train[i][1])

X_test = []
y_test = []
for i in range(len(test)):
	X_test.append(test[i][0])
	y_test.append(test[i][1])

del train
del test # liberate more RAM

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

print("len trains",len(X_train),len(y_train))
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

print("len pre reshape",len(X_train))
print(X_train[0])


X_train = X_train.reshape(X_train.shape[0], inp_dim, inp_dim,1)
# unselect this for 3d images
#X_train = X_train.reshape(X_train.shape[0], inp_dim, inp_dim, inp_dim,1)

print("len post reshape",len(X_train))
print(X_train[0])

X_test = X_test.reshape(X_test.shape[0], inp_dim, inp_dim, 1)
input_shape = (inp_dim, inp_dim, 1)



# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# convert class vectors to binary class matrices (2 <-> binary classes)
Y_train = np_utils.to_categorical(y_train, 2)
Y_test = np_utils.to_categorical(y_test, 2)

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


model = Sequential()


model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))

model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch)
model.save("../models/model_0.mdl")


# model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#           verbose=1, validation_data=(X_test, Y_test))




score = model.evaluate(X_test, Y_test, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])




