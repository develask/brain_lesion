import os
from keras.models import load_model
import nibabel as nib
import numpy as np
import math
import brain as br


modelo_file = "../models/model_0.mdl"
result_path = "../results/new_image2.nii.gz"

model = load_model(modelo_file)

brain_id = "tka004"

inp_dim = 19
img_types = ["flair"]
sample_type = "2dx"

brain = br.Brain(brain_id)
brain.createSlices(step=1)
brain.split(1)

total = brain.getData(img_types, sample_type, inp_dim)
x = total[0][0]
y_real = total[0][1]

y_pred = model.predict(x)
y_pred = y_pred[:,1] >= 0.5

brain.train = np.concatenate((
	brain.train,
	y_pred.T
), axis=1)

new_im = np.zeros(brain.mask)
for a in brain.train:
	new_im[a[0],a[1],a[2]] = a[4]
	

new_image = nib.Nifti1Image(new_im,  np.eye(4))
nib.save(new_image, result_path)




