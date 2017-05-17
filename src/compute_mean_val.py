import numpy as np
import nibabel as nib
import os

std = mask = nib.load("../data/standars/MNI152_T1_1mm_first_brain_mask.nii.gz").get_data()>0

total_pixels = 0
positives = 0


for mask in os.walk("../data/mask/normalized/"):
	direcs = mask
	break
masks = direcs[2]

for mask in masks:
	img = nib.load("../data/mask/normalized/" + mask).get_data()
	brain = img * std
	total_pixels += brain.shape[0] *  brain.shape[1] *  brain.shape[2]
	positives += np.sum(brain)
	if "002" in mask:
		print(mask,np.sum(brain),brain.shape[0] *  brain.shape[1] *  brain.shape[2])

print("total",total_pixels)
print("positive",positives)
print("inverted balance rate",total_pixels/positives)


# total 129978576
# positive 618424
# inverted balance rate 210.177121198
