import os
import image_functions as imf
from keras.models import load_model
import sample_cretation as sc
import nibabel as nib
import numpy as np
import math


modelo_file = "../models/model_0.mdl"

model = load_model(modelo_file)

inp_dim = 19
step = 1
img_types = ["flair"]
sample_type = "2dx"

cerebro = "tka004"
print("Calculando parte del cerebro")
ex = sc.Examples()
ex.initilize(cerebro)
ex.get_examples(step = step,output_type="classes")
ex.getData([], img_types, "2dx", inp_dim, crbs = cerebro)

brainim = nib.load("../data/standars/MNI152_T1_1mm_first_brain_mask.nii.gz")
brain = imf.OurImage(brainim.get_data(), "brain")
x = ex.getData([ex[0][0]], img_types, "2dx", inp_dim, crbs = cerebro)
x = np.asarray(x)
x.reshape(x.shape[0], inp_dim, inp_dim,len(img_types))
print("Emaitza")
total = len(ex.pairs)
start = 0
step = math.floor(total/100)
stop = step-1
print(stop*100/float(total),"%     ",stop," - ",total, end="\r",flush=True)
while True:
	l=[]
	for index in range(start, stop):
		l.append(ex[index][0])
	x = ex.getData(l, img_types, "2dx", inp_dim, crbs = cerebro)
	x = np.asarray(x)
	x = x.reshape(x.shape[0], inp_dim, inp_dim,len(img_types))
	x = model.predict(x)
	i=0
	for a in x:
		y_pred = a[1] >= 0.5
		brain.data[l[i].x][l[i].y][l[i].z] = float(1 if y_pred else 0)
		i+=1
	print(stop*100/float(total),"%     ",stop," - ",total, end="\r",flush=True)
	start+=step
	stop+=step
	if (stop>total):
		stop=total
	if(start>total):
		break

print("end                               ")

new_image = nib.Nifti1Image(brain.data,  np.eye(4))
nib.save(new_image, 'new_image2.nii.gz')




