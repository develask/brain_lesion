import numpy as np
import nibabel as nib

import sys

pred = sys.argv[2]

mask = nib.load("../data/standars/MNI152_T1_1mm_first_brain_mask.nii.gz").get_data()>0
original = nib.load("../data/mask/normalized/"+pred+"_mask_norm.nii.gz").get_data()>0
predicted = nib.load(sys.argv[1]).get_data()>0

total = np.sum(mask)

TP = np.sum(mask * original * predicted)
P = np.sum(mask * original)

TN = np.sum(mask * (1 - original) * (1 - predicted))
N = np.sum(mask * (1 - original))


print("TPR:", float(TP)/P)
print("TNR:", float(TN)/N)
print("Total positives:", P)
print("Total negatives:", N)
