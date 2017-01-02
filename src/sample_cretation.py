import os
import numpy as np
import nibabel as nib
import image_functions as imf

class Examples():
	def __init__(self):
		self.pairs = []

	def remove_elem(self,ind):
		self.pairs.pop(ind) #removing its index

	
	def get_examples(self, img_type,sample_type,dim,step = 1,output_type="regression"):
		# img_type, e.g. FA, MO,...
		# sample_type, e.g. 2dx, 3d...
		# dim dimension of the subsample

		# this function loads both the inputs (the subsamples) and the
		# desired outputs (the mask)
		for f in os.walk("../data/"+img_type+"/normalized/"):
			direcs = f
			break
		images_in = direcs[2]

		for f in os.walk("../data/mask/normalized/"):
			direcs = f
			break
		images_ou = direcs[2]

		images_inp = []
		images_out = []
		for im in images_in:
			for im2 in images_ou:
				if im[0:6] == im2[0:6]:
					images_inp.append(im)
					images_out.append(im2)

		# print(images_inp)
		# print(images_out)



		for i in range(len(images_inp)):
			print("starting with a new image..")
			img_in = nib.load("../data/"+img_type+"/normalized/" + images_inp[i])
			data_in = imf.OurImage(img_in.get_data())

			img_out = nib.load("../data/mask/normalized/" + images_out[i])
			data_out = imf.OurImage(img_out.get_data())

			slices = data_in.get_slices(dim, sample_type, step)
			print(len(slices))
			for j in range(len(slices)):
				if output_type == "regression":
					self.pairs.append((slices[j].data, ######## akaso solo slices[j],
													   # para pasar info adicional a la NN
									   data_out[slices[j].x][slices[j].y][slices[j].z]))
				else:
					out = data_out[slices[j].x][slices[j].y][slices[j].z]
					klasea = ()
					if out > 0:
						klasea = (0,1)
					else:
						klasea = (1,0)

					self.pairs.append((slices[j].data, ######## akaso solo slices[j],
													   # para pasar info adicional a la NN
									   klasea))					













		









