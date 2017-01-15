import numpy as np

class Slice():
	def __init__(self, x,y,z,fromIm):
		self.x = x
		self.y = y
		self.z = z
		self.fromIm = fromIm

class OurImage():
	def __init__(self, data, name):
		self.data = np.asarray(data, dtype=np.float32)
		self.lenx = len(data)
		self.leny = len(data[0])
		self.lenz = len(data[0][0])
		self.name = name

	def __iter__(self):
		for i in range(self.lenx):
			for j in range(self.leny):
				for k in range(self.lenz):
					yield self.data[i][j][k]

	def __getitem__(self, num):
		return self.data[num]

	def slice_matrix(self,x,y,z,dim,tipo):
		# tipo = 2dx, 2dy, 2dz, 3d
		# dim ha de ser impar
		padding = int((dim-1)/2)
		if tipo == "2dx":
			dato = np.full((dim,dim), self.data[0][0][0])
			for i in range(dim):
				for j in range(dim):
					if 0 <= y-padding+i < self.leny and 0 <= z-padding+j < self.lenz:
						dato[i][j] = self.data[x][y-padding+i][z-padding+j]

		elif tipo == "2dy":
			dato = np.full((dim,dim), self.data[0][0][0])
			for i in range(dim):
				for j in range(dim):
					if 0 <= x-padding+i < self.lenx and 0 <= z-padding+j < self.lenz:
						dato[i][j] = self.data[x-padding+i][y][z-padding+j]

		elif tipo == "2dz":
			dato = np.full((dim,dim), self.data[0][0][0])
			for i in range(dim):
				for j in range(dim):
					if 0 <= x-padding+i < self.lenx and 0 <= y-padding+j < self.leny:
						dato[i][j] = self.data[x-padding+i][y-padding+j][z]

		elif tipo == "3d":
			dato = np.full((dim,dim,dim), self.data[0][0][0])
			for i in range(dim):
				for j in range(dim):
					for k in range(dim):
						if 0 <= x-padding+i < self.lenx and 0 <= y-padding+j < self.leny and  0 <= z-padding+k <= self.lenz:
							dato[i][j][k] = self.data[x-padding+i][y-padding+j][z-padding+k]
		else:
			#print("Wrong Sample Type:", tipo)
			raise Exception("Wrong Sample Type: "+tipo)
		return dato


	def get_slice(self,x,y,z):
		return(Slice(x,y,z,self.name))

	def filterByImage(self, image, step = 1):
		if (image.lenx != self.lenx or image.leny != self.leny or image.lenz != self.lenz):
			raise Exception("different image sizes")
		slices = []
		out_value = image[0][0][0]
		for i in range(0,self.lenx, step):
			for j in range(0,self.leny, step):
				for k in range(0,self.lenz, step):
					if (image[i][j][k] != out_value):
						slices.append(self.get_slice(i,j,k))
		return(slices)

