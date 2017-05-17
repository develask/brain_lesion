import os
from keras.models import load_model
import nibabel as nib
import numpy as np
import math
import brain as br


modelo_file = "../models/model_paralel_v1_0.mdl"
result_path = "../results/tka002_paralel_v1_0.nii.gz"


model = load_model(modelo_file)

brain_id = "tka007" #1
# "tka015" #2
# tka018 #3
# tka021 #x
modelo_file = "../models/model_paralel_v1_1.mdl"
result_path = "../results/"+brain_id+"_paralel_v1_1.nii.gz"

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
print(fin*100/float(final),"%                 ",fin," / ", final, end="\r", flush=True)
#print(fin*100/float(final),"%                 ",fin," / ", final)
new_im = np.zeros(brain.mask.shape)
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
	y_pred = y_pred[:,1] >= 0.5
	brain.train = np.concatenate((
		brain.train,
		y_pred[:, np.newaxis]
	), axis=1)
	for a in brain.train:
		new_im[a[0],a[1],a[2]] = a[4]
	print(fin*100/float(final),"%                 ",fin," / ", final, end="\r", flush=True)
	#print(fin*100/float(final),"%                 ",fin," / ", final)
	inicio += step
	fin += step
	if fin>final:
		fin = final
	if inicio > final:
		break
new_image = nib.Nifti1Image(new_im,  np.eye(4))
nib.save(new_image, result_path)




