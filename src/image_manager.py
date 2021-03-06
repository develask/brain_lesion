import brain as br
import numpy as np
import sys
#from psutil import virtual_memory

class ImageManager():
	def __init__(self):
		self.images = []
		self.mem = 0

	def reset(self):
		self.images = []
		self.mem = 0

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

	def joinSlices(self):
		self.r = None
		for index in range(len(self.images)):
			if self.r == None:
				self.r = self.images[index].result
				self.r = np.insert(self.r, 4, index, axis=1)
			else:
				tmp = self.images[index].result
				tmp = np.insert(tmp, 4, index, axis=1)
				self.r = np.concatenate((self.r, tmp), axis=0)
			self.images[index].result = []
			self.images[index].train = []
			self.images[index].test = []
		np.random.shuffle(self.r)
		self.start = 0

	def prepareNewBatch(self, num):
		r = self.r[self.start:self.start+num]
		if r.shape[0] == 0:
			raise Exception("End of Epoch")
		self.start += num
		for index in range(len(self.images)):
			tmp = r[r[:,4]==index]
			self.images[index].train = tmp
			self.images[index].test = tmp[tmp.shape[0]:]


	def getData(self, img_types, sample_type, dim, p=True):
		# self.memoryAvailable(img_types, sample_type, dim,p)
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
		tmp = 1-train_y
		tmp = tmp[:, np.newaxis]
		tmp2 = train_y[:, np.newaxis]
		train_y = np.concatenate((tmp, tmp2), axis=1)
		tmp = 1-test_y
		tmp = tmp[:, np.newaxis]
		tmp2 = test_y[:, np.newaxis]
		test_y = np.concatenate((tmp, tmp2), axis=1)
		return([(train_x,train_y),(test_x,test_y)])

	# def memoryAvailable(self, img_types, sample_type, dim, p = True):
	# 	size = 0
	# 	size_t = sys.getsizeof(np.array([0.5]))
	# 	for cr in self.images:
	# 		size += cr.train.shape[0] + cr.test.shape[0]
	# 	size *= dim
	# 	size *= dim
	# 	size *= len(img_types)
	# 	size *= size_t

	# 	max_ram = 2 * 1024 * 1024 * 1024# 2GB ram
	# 	mem = virtual_memory()
	# 	mem = mem.total - max_ram
	# 	self.mem += size
	# 	if p:
	# 		print("Needed Memory:", self.mem/(1024**3))
	# 		print("Avilable Memory:", mem/(1024**3))
	# 	if self.mem>mem:
	# 		raise Exception("Need more memory: (", self.mem,"/",mem,")")













