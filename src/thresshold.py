import nibabel as nib
import numpy as np
import sys


image = sys.argv[1]
image_r = sys.argv[2]
t = sys.argv[3]

i = nib.load(image).get_data()
i = i >= float(t)
i.astype(int)
new_image = nib.Nifti1Image(i,  np.eye(4))
nib.save(new_image, image_r)