import os
import numpy as np
import nibabel as nib
import sys

class Slice():
	def __init__(self, x,y,z,data,tipo):
		self.x = x
		self.y = y
		self.z = z
		self.data = data
		self.tipo = tipo  
		

def get_slice(data,x,y,z,dim,tipo):
	# tipo = 2dx, 2dy, 2dz, 3d
	# dim ha de ser impar
	padding = int((dim-1)/2)
	if tipo == "2dx":
		dato = np.zeros((dim,dim))
		for i in range(dim):
			for j in range(dim):
				if not(y-padding+i<0 or z-padding+j<0):
					try:
						dato[i][j] = data[x][y-padding+i][z-padding+j]
					except:
						pass

	elif tipo == "2dy":
		dato = np.zeros((dim,dim))
		for i in range(dim):
			for j in range(dim):
				if not(x-padding+i<0 or z-padding+j<0):
					try:
						dato[i][j] = data[x-padding+i][y][z-padding+j]
					except:
						pass

	elif tipo == "2dz":
		dato = np.zeros((dim,dim))
		for i in range(dim):
			for j in range(dim):
				if not(x-padding+i<0 or z-padding+j<0):
					try:
						dato[i][j] = data[x-padding+i][y-padding+j][z]
					except:
						pass

	elif tipo == "3d":
		dato = np.zeros((dim,dim,dim))
		for i in range(dim):
			for j in range(dim):
				for k in range(dim):
					if not(x-padding+i<0 or y-padding+j<0 or z-padding+k<0):
						try:
							dato[i][j][k] = data[x-padding+i][y-padding+j][z-padding+k]
						except:
							pass	

	else:
		print("Wrong tipo:",tipo)
		return(None)				

	return(Slice(x,y,z,dato,tipo))

x = 0
data = [[[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]],[[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]],[[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,195,240],[21,22,23,44,45]]]


print(get_slice(data,1,4,4, 3,"3d").data)


quit()
dir_name = sys.argv[1]
for archivo in os.walk("../data/raw/"+dir_name):
	direcs = archivo
	break
images = direcs[2]

for image in images:
	print("IMAGE " + image)
	try:
		print("i'm trying...")
		img = nib.load("../data/raw/" + dir_name + "/" + image)
		print("try finished")

		data = img.get_data()
		std_data(data)
		img = nib.Nifti1Image(data, np.eye(4))
		tipo = getImgType(image)
		#nib.save(img,"../data/tipo/"+image[0:-8]+"_norm.nii.gz")
		print("../data/" + tipo + "/" + image[0:-7] + "_norm.nii.gz")
		nib.save(img, "../data/" + tipo + "/normalized/" + image[0:-7] + "_norm.nii.gz")
	except:
		print("EOFError in IMAGE:",image)
		print("passing...")
		continue


