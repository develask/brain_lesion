import numpy as np
import nibabel as nib

pred = 
real = 

mask = nib.load("../data/standars/MNI152_T1_1mm_first_brain_mask.nii.gz").getdata()>0
original = nib.load("../data/mask/normalized/"+pred).getdata()>0
predicted = nib.load("../results/"+real).getdata()>0

total = np.sum(mask)

TP = np.sum(mask * original * predicted)
P = np.sum(mask * original)

TN = np.sum(mask * (1 - original) * (1 - predicted))
N = np.sum(mask * (1 - original))


print("TPR:", TP/P)
print("TNR:", TN/N)
print("Total positives:", P)
print("Total negatives:", N)
