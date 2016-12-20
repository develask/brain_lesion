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

#############################################

####### LOAD DATA ##########

batch_size = 128
nb_classes = 2
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

ex = sc.Examples()
ex.get_examples("flair","2dx",9,step = 3,output_type="classes")


print(ex.pairs[322])
print("number of examples",len(ex.pairs))

k = 0
for x in ex.pairs:
	if x[1][1] == 1:
		k += 1

print("number of positive after selection: ",k)


img_out = nib.load("../data/mask/normalized/tka003_lesion_mask_norm.nii.gz")
data_out = imf.OurImage(img_out.get_data())

k=0
for x in data_out:
	if x >0:
		k+=1

img_out = nib.load("../data/mask/normalized/tka004_lesion_mask_norm.nii.gz")
data_out = imf.OurImage(img_out.get_data())


for x in data_out:
	if x >0:
		k+=1

print("number of positive before selection: ",k)

quit()




# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# if K.image_dim_ordering() == 'th':
#     X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#     X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else: # indent the next to fit the if-else structure if needed....
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

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

print("gonna train")


model.fit(X_train, Y_train,
batch_size=batch_size, nb_epoch=nb_epoch)


# model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#           verbose=1, validation_data=(X_test, Y_test))




score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])




