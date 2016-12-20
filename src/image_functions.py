import numpy as np

class Slice():
	def __init__(self, x,y,z,data,tipo):
		self.x = x
		self.y = y
		self.z = z
		self.data = data
		self.tipo = tipo

class OurImage():
	def __init__(self, data):
		self.data = data
		self.lenx = len(data)
		self.leny = len(data[0])
		self.lenz = len(data[0][0])

	def __iter__(self):
		for i in range(self.lenx):
			for j in range(self.leny):
				for k in range(self.lenz):
					yield self.data[i][j][k]

	def __getitem__(self, num):
		return self.data[num]

	def get_slice(self,x,y,z,dim,tipo):
		# tipo = 2dx, 2dy, 2dz, 3d
		# dim ha de ser impar
		padding = int((dim-1)/2)
		if tipo == "2dx":
			dato = np.full((dim,dim), self.data[0][0][0])
			for i in range(dim):
				for j in range(dim):
					if 0 <= y-padding+i <= self.leny and 0 <= z-padding+j <= self.lenz:
						dato[i][j] = self.data[x][y-padding+i][z-padding+j]

		elif tipo == "2dy":
			dato = np.full((dim,dim), self.data[0][0][0])
			for i in range(dim):
				for j in range(dim):
					if 0 <= x-padding+i <= self.lenx and 0 <= z-padding+j <= self.lenz:
						dato[i][j] = self.data[x-padding+i][y][z-padding+j]

		elif tipo == "2dz":
			dato = np.full((dim,dim), self.data[0][0][0])
			for i in range(dim):
				for j in range(dim):
					if 0 <= x-padding+i <= self.lenx and 0 <= y-padding+j <= self.leny:
						dato[i][j] = self.data[x-padding+i][y-padding+j][z]

		elif tipo == "3d":
			dato = np.full((dim,dim,dim), self.data[0][0][0])
			for i in range(dim):
				for j in range(dim):
					for k in range(dim):
						if 0 <= x-padding+i <= self.lenx and 0 <= y-padding+j <= self.leny and  0 <= z-padding+k <= self.lenz:
							dato[i][j][k] = self.data[x-padding+i][y-padding+j][z-padding+k]

		else:
			#print("Wrong Sample Type:", tipo)
			raise Exception("Wrong Sample Type: "+tipo)

		return(Slice(x,y,z,dato,tipo))

	def get_slices(self, dim, sample_type, step = 1):
		slices = np.zeros(int((self.lenx+step-1)/step) * int((self.leny+step-1)/step) * int((self.lenz+step-1)/step))
		indice = 0
		for i in range(self.lenx, step):
			for j in range(self.leny, step):
				for k in range(self.lenz, step):
					slices[indice] = self.get_slice(i,j,k,dim,sample_type)
					indice+=1
		return slices

