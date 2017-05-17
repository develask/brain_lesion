import numpy as np
import nibabel as nib
import os

class Brain():
	def __init__(self, brain):
		self.name = brain
		self.mask = nib.load("../data/mask/normalized/" + brain + "_mask_norm.nii.gz").get_data()
		self.lenx = len(self.mask)
		self.leny = len(self.mask[0])
		self.lenz = len(self.mask[0][0])
		self.types = {}

	def load(self, newtype):
		if not newtype in self.types:
			for f in os.walk("../data/"+newtype+"/normalized/"):
				direcs = f
				break
			images = direcs[2]
			for i in range(len(images)):
				name = images[i]
				try:
					if name.index(self.name)>-1:
						self.types[newtype] = nib.load("../data/"+newtype+"/normalized/" + name).get_data()
				except:
					pass

	def createSlices(self, step):
		standar = nib.load("../data/standars/MNI152_T1_1mm_first_brain_mask.nii.gz").get_data()
		indexes = np.indices((int(self.lenx/step), int(self.leny/step), int(self.lenz/step)))
		indexes *= step
		ran = np.random.randint(step, size=indexes.shape)

		indexes += ran

		length = indexes[0].shape[0]*indexes[0].shape[1]*indexes[0].shape[2]

		i1 = indexes[0].reshape(length)
		i2 = indexes[1].reshape(length)
		i3 = indexes[2].reshape(length)
		tmp = self.mask[i1,i2,i3]

		res = standar[i1,i2,i3] > 0
		i1 = i1[res]
		i2 = i2[res]
		i3 = i3[res]
		tmp = tmp[res]
		length = tmp.shape[0]

		self.result = np.concatenate((
			i1.reshape(length,1),
			i2.reshape(length,1),
			i3.reshape(length,1),
			tmp.reshape(length,1)
		), axis=1)

	def balance(self, bal):
		pos = self.result[self.result[:,3]==1]
		neg = self.result[self.result[:,3]==0]
		neg = neg[:pos.shape[0]*bal]
		pos = np.concatenate((pos,neg), axis=0)
		np.random.shuffle(pos)
		self.result = pos

	def split(self, portion):
		split_num = int(portion * self.result.shape[0])
		self.train = self.result[:split_num]
		self.test = self.result[split_num:]

	def getData(self, img_types, sample_type, dim):
		result_train = None
		result_test = None
		for img_type in img_types:
			self.load(img_type)
			margin = np.full((self.mask.shape[0]+dim-1, self.mask.shape[1]+dim-1, self.mask.shape[2]+dim-1), self.mask[0,0,0])
			mar = int((dim-1)/2)
			margin[mar:-mar, mar:-mar, mar:-mar] = self.types[img_type][:,:,:]
			if sample_type == "2dx":
				train_X = np.empty((self.train.shape[0], dim, dim))
				i = 0
				for x in self.train:
					train_X[i,:,:] = margin[x[0]+mar-1,x[1]:x[1]+dim,x[2]:x[2]+dim]
					i+=1
				test_X = np.empty((self.test.shape[0], dim, dim))
				i = 0
				for x in self.test:
					test_X[i,:,:] = margin[x[0]+mar-1,x[1]:x[1]+dim,x[2]:x[2]+dim]
					i+=1
			elif sample_type == "2dy":
				train_X = np.empty((self.train.shape[0], dim, dim))
				i = 0
				for x in self.train:
					train_X[i,:,:] = margin[x[0]:x[0]+dim,x[1]+mar-1,x[2]:x[2]+dim]
					i+=1
				test_X = np.empty((self.test.shape[0], dim, dim))
				i = 0
				for x in self.test:
					test_X[i,:,:] = margin[x[1]:x[1]+dim,x[0]+mar-1,x[2]:x[2]+dim]
					i+=1
			elif sample_type == "2dz":
				train_X = np.empty((self.train.shape[0], dim, dim))
				i = 0
				for x in self.train:
					train_X[i,:,:] = margin[x[0]:x[0]+dim,x[1]:x[1]+dim,x[2]+mar-1]
					i+=1
				test_X = np.empty((self.test.shape[0], dim, dim))
				i = 0
				for x in self.test:
					test_X[i,:,:] = margin[x[1]:x[1]+dim,x[2]:x[2]+dim,x[0]+mar-1]
					i+=1
			elif sample_type == "3d":
				train_X = np.empty((self.train.shape[0], dim, dim, dim))
				i = 0
				for x in self.train:
					train_X[i,:,:,:] = margin[x[0]:x[0]+dim,x[1]:x[1]+dim,x[2]:x[2]+dim]
					i+=1
				test_X = np.empty((self.test.shape[0], dim, dim, dim))
				i = 0
				for x in self.test:
					test_X[i,:,:,:] = margin[x[0]:x[0]+dim,x[1]:x[1]+dim,x[2]:x[2]+dim]
					i+=1
			if type(result_train) == type(None):
				if sample_type == "3d":
					result_train = train_X[:,:,:,:, np.newaxis]
					result_test = test_X[:,:,:,:, np.newaxis]
				else:
					result_train = train_X[:,:,:, np.newaxis]
					result_test = test_X[:,:,:, np.newaxis]
			else:
				result_train=np.insert(result_train, result_train.shape[-1], train_X, axis=len(result_train.shape)-1)
				result_test=np.insert(result_test, result_test.shape[-1], test_X, axis=len(result_test.shape)-1)
		return([(result_train, self.train[:,3]), (result_test, self.test[:,3])])


