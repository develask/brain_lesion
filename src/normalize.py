import os
import numpy as np
import nibabel as nib
import sys

import re
from copy import deepcopy
import image_functions as imf


#img = nib.load("datos/004/tka004_lesion_mask.nii.gz")
#data = img.get_data()

#img = nib.load("datos/003/tka003_dwi_MD.nii.gz")
#data2 = img.get_data()


#print("V1")
#print(len(data),len(data[0]),len(data[0][0]))

#print("MD")
#print(len(data2),len(data2[0]),len(data2[0][0]))

#quit()

#data = data[126]

#img = nib.Nifti1Image(data, np.eye(4))
#nib.save(img,"datos/003/tka003_dwi_L3_duplicado_2D.nii.gz")

#,quit()

def std_data(data,valor_min=0,valor_max=1):
	max_pre = np.max(data) # valor maximo en data
	min_pre = np.min(data)

	len_1 = len(data)
	len_2 = len(data[0])
	len_3 = len(data[0][0])

	for i in range(len_1):
		for j in range(len_2):
			for k in range(len_3):
				data[i][j][k] = (float(data[i][j][k]) - (min_pre - valor_min)) * (valor_max-valor_min) / (max_pre-min_pre)

ind=0
def getImgType(filename):
	#tipo = ""
	#for i in range(len(filename)):
	#	if filename[i]=="_":
	#		ind=i
	#i = ind
	#while filename[i] != ".":
	#	 tipo += filename[i]
	#	i += 1
	#return(tipo)

	#mask = '_[a-zA-Z]+\.nii.gz'
	#result = re.search(mask).match(filename)
	#mask = 'nii.gz'
	#p = re.compile(mask);
	#result = p.match(filename)
	result = re.search(".*\_([a-zA-Z0-9]+)\.nii\.gz",filename).group(1)
	print(result)
	return (result)

#std_data(data)

#print(np.min(data))
#print(np.max(data))	

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
		data_tmp = img.get_data()
		data = imf.OurImage(data_tmp)
		std_data(data.data)
		img = nib.Nifti1Image(data, np.eye(4))
		tipo = getImgType(image)
		#nib.save(img,"../data/tipo/"+image[0:-8]+"_norm.nii.gz")
		print("../data/" + tipo + "/" + image[0:-7] + "_norm.nii.gz")
		nib.save(img, "../data/" + tipo + "/normalized/" + image[0:-7] + "_norm.nii.gz")
	except:
		print("EOFError in IMAGE:",image)
		print("passing...")
		continue


