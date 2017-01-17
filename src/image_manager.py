import brain as br
import numpy as np

class ImageManager():
	def __init__(self):
		self.images = []

	def reset(self):
		self.images = []

	def init(self, images):
		for im in images:
			self.images.append(br.Brain(im))

	def createSlices(self, step = 2):
		for im in self.images:
			im.createSlices(step)

	def balance(self, bal = 10):
		for im in self.images:
			im.balance(bal)

	def split(self, portion=0.8):
		for im in self.images:
			im.split(portion)

	def getData(self, img_types, sample_type, dim):
		train_x = None
		train_y = None
		test_x = None
		test_y = None
		for im in self.images:
			tot = im.getData(img_types, sample_type, dim)
			if (type(train_x) == type(None)):
				train_x = tot[0][0]
				train_y = tot[0][1]
				test_x = tot[1][0]
				test_y = tot[1][1]
			else:
				train_x = np.concatenate((train_x, tot[0][0]), axis=0)
				train_y = np.concatenate((train_y, tot[0][1]), axis=0)
				test_x = np.concatenate((test_x, tot[1][0]), axis=0)
				test_y = np.concatenate((test_y, tot[1][1]), axis=0)
		return([(train_x,train_y),(test_x,test_y)])

