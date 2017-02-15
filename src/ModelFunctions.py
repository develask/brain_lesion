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
	def __init__(self, model, dataManager):
		self.model = model
		self.dataManager = dataManager
		self.brains = ["tka002","tka003","tka004","tka005","tka006","tka007","tka009","tka010","tka011","tka012","tka013","tka015","tka016","tka017","tka018","tka019","tka020","tka021"]


	def leaveOneOut(self, nb_epoch = 250, batch_size = 128):

		init_ler = 0.05
		final_ler = 0.005
		dec = (final_ler/init_ler)**(1/nb_epoch)


		for i in range(len(self.brains)):
			test = [self.brains[i]]
			train = self.brains[0:i] + self.brains[i+1:len(self.brains)]

			self.dataManager.setTrain(train)

			ler = init_ler
			for j in range(nb_epoch):
				print("Starting epoch:", j+1, "/", nb_epoch)
				print("Genrating new training data")
				
				d = self.dataManager.getData()

				sgd = SGD(lr=ler,decay=0,momentum=0.0,nesterov = False)
				self.model.compile(loss='binary_crossentropy',
			              			optimizer=sgd,
			              				metrics=['accuracy'])
				tr_h = self.model.fit(d[0], d[1], batch_size=batch_size, nb_epoch=1,verbose=2)
				print("train_loss", tr_h.history["loss"][0])
				cv_history.append(tr_h.history["loss"][0])
				ler *= dec

			self.model.save("../models/model_" + model_name +"_"+ str(i) + ".mdl")

class DataManager():
	def __init__(self):
		self.tr = imm.ImageManager()

	def setTrain(self, brains):
		self.tr.reset()
		self.tr.init(brains)

	def getData(self):	
		self.tr.createSlices(step=step)
		self.tr.balance(bal_train)
		self.tr.split(1)
		X_train_x = self.tr.getData(img_types, "2dx", inp_dim_2d)[0]
		y_train = X_train_x[1]
		X_train_x = X_train_x[0]
		X_train_y = self.tr.getData(img_types, "2dy", inp_dim_2d)[0][0]
		X_train_z = self.tr.getData(img_types, "2dz", inp_dim_2d)[0][0]
		X_train_3d = self.tr.getData(img_types, "3d", inp_dim_3d)[0][0]

		return(([X_train_x,X_train_y, X_train_z, X_train_3d], y_train))
