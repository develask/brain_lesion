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
import time

import tensorflow as tf
import json

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


class Model():
	def __init__(self, model_name, model, getDataFunc):
		self.model_name = model_name
		self.model = model
		self.getDataFunc = getDataFunc
		self.brains = ["tka003","tka004","tka005","tka006","tka007","tka009","tka010","tka011","tka012","tka013","tka015","tka016","tka017","tka018","tka019","tka020","tka021","tka002"]


	def leaveOneOut(self, nb_epoch = 250, batch_size = 128, init_ler = 0.05, final_ler = 0.005):
		dec = (final_ler/init_ler)**(1/nb_epoch)

		traindata = DataManager(self.getDataFunc)

		for i in range(len(self.brains)):
			test = [self.brains[i]]
			train = self.brains[0:i] + self.brains[i+1:len(self.brains)]
			print("Starting cv number", i, "out of", len(self.brains))
			traindata.setTrain(train)

			cv_history = []

			ler = init_ler
			start_time = time.time()
			for j in range(nb_epoch):
				print("Starting epoch:", j+1, "/", nb_epoch)
				print("Genrating new training data")
				
				d = traindata.getData()

				sgd = SGD(lr=ler,decay=0,momentum=0.0,nesterov = False)
				self.model.compile(loss='binary_crossentropy',
			              			optimizer=sgd,
			              				metrics=['accuracy'])
				tr_h = self.model.fit(d[0], d[1], batch_size=batch_size, nb_epoch=1,verbose=2)
				print("train_loss", tr_h.history["loss"][0])
				cv_history.append(tr_h.history["loss"][0])
				ler *= dec
			elapsed_time = time.time() - start_time
			print("#################################################")
			print("\tTime training (",i,"):", elapsed_time)
			print("#################################################")
			self.model.save("../models/model_" + self.model_name +"_for_"+ test[0] + ".mdl")
			with open("../models/hist_"+ self.model_name +"_for_"+ test[0] +".json","w") as tf:
				tf.write(json.dumps(cv_history))
			break		


class DataManager():
	def __init__(self, datafunc):
		self.tr = imm.ImageManager()
		self.datafunc = datafunc

	def setTrain(self, brains):
		self.tr.reset()
		self.tr.init(brains)

	def getData(self):
		return self.datafunc(self.tr)


		
