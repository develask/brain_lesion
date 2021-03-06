import os
from keras.models import load_model
import nibabel as nib
import numpy as np
import math
import brain as br

import sys

brain_id = sys.argv[2]
#"tka007" #1
# "tka015" #2
# tka018 #3
# tka021 #x
modelo_file = sys.argv[1]

model = load_model(modelo_file)

nombre = modelo_file.split("/")[-1].split(".")[0]

result_path = "../results/"+brain_id+"_with_"+nombre+".nii.gz"

inp_dim_2d = 35
inp_dim_3d = 11
img_types = ["flair","FA","anatomica"]

brain = br.Brain(brain_id)
brain.createSlices(step=1)
#brain.split(1)
final = brain.result.shape[0]
inicio = 0
fin = int(final*0.01)
step = fin
np.random.shuffle(brain.result)
#print(fin*100/float(final),"%                 ",fin," / ", final, end="\r", flush=True)
print(fin*100/float(final),"%                 ",fin," / ", final)
new_im = np.zeros(brain.mask.shape, dtype="float32")
while True:
	brain.train = brain.result[inicio:fin]
	brain.test = brain.result[0:0]


	total = brain.getData(img_types, "2dx", inp_dim_2d)
	x_x = total[0][0]
	x_y = brain.getData(img_types, "2dy", inp_dim_2d)[0][0]
	x_z = brain.getData(img_types, "2dz", inp_dim_2d)[0][0]
	x_3d = brain.getData(img_types, "3d", inp_dim_3d)[0][0]
	y_real = total[0][1]
	y_pred = model.predict([x_x,x_y,x_z,x_3d])

	print(y_pred)
	y_pred = y_pred[:,1]
	print(np.mean(y_pred), np.max(y_pred))
	brain.train = np.concatenate((
		brain.train,
		y_pred[:, np.newaxis]
	), axis=1)
	for a in brain.train:
		new_im[int(a[0]),int(a[1]),int(a[2])] = a[4]
	#print(fin*100/float(final),"%                 ",fin," / ", final, end="\r", flush=True)
	print(fin*100/float(final),"%                 ",fin," / ", final)
	inicio += step
	fin += step
	if fin>final:
		fin = final
	if inicio > final:
		break
new_image = nib.Nifti1Image(new_im,  np.eye(4))
nib.save(new_image, result_path)




