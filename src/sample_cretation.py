import os
import numpy as np
import nibabel as nib
import image_functions as imf
import random as rdm

class Examples():
	def __init__(self):
		self.pairs = []

	def remove_elem(self,ind):
		self.pairs.pop(ind) #removing its index

	def shuffle_exs(self):
		rdm.shuffle(self.pairs)

	def reset_exs(self):
		del self.pairs
		self.pairs = []

	
	def get_examples(self, img_type,sample_type,dim,step = 1,output_type="regression"):
		# img_type, e.g. FA, MO,...
		# sample_type, e.g. 2dx, 3d...
		# dim dimension of the subsample

		# this function loads both the inputs (the subsamples) and the
		# desired outputs (the mask)
		brain = imf.OurImage(nib.load("../data/standars/MNI152_T1_1mm_first_brain_mask.nii.gz").get_data())

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

			#slices = data_in.get_slices(dim, sample_type, step)
			slices = data_in.filterByImage(brain, dim, sample_type, step)
			print(len(slices))
			for j in range(len(slices)):
				if output_type == "regression":
					self.pairs.append((slices[j], ######## akaso solo slices[j],
													   # para pasar info adicional a la NN
									   data_out[slices[j].x][slices[j].y][slices[j].z]))
				else:
					out = data_out[slices[j].x][slices[j].y][slices[j].z]
					klasea = ()
					if out > 0:
						klasea = 1
						#klasea = (0,1)
					else:
						# klasea = (1,0)
						klasea = 0
					self.pairs.append((slices[j], ######## akaso solo slices[j],
													   # para pasar info adicional a la NN
									   klasea))

	def valance(self, portion):
		# portion --> neg/positive rate

		num_pos = 0
		for pair in self.pairs:
			if pair[1] > 0:
				num_pos += 1

		r = list(range(0,len(self.pairs)))
		rdm.shuffle(r)

		num_pos = num_pos * portion
		num_neg = 0
		to_be_removed = [] # list of index that should be removed
		for i in r:
			if self.pairs[i][1] == 0:
				if num_neg < num_pos:
					num_neg += 1
				else:
					to_be_removed.append(i)

		to_be_removed = sorted(to_be_removed,reverse=True)
		for i in to_be_removed:
			self.remove_elem(i)


	def split(self, portion):

		self.shuffle_exs() ## shuffle the training - test examples

		tot = portion*len(self.pairs)
		i = 0

		X_train = []
		y_train = []

		X_test = []
		y_test = []


		for pair in self.pairs:
			if i<tot:
				X_train.append(pair[0].getData())
				y_train.append(pair[1])
				i += 1
			else:
				X_test.append(pair[0].getData())
				y_test.append(pair[1])
		return [(X_train,y_train),(X_test,y_test)]












		









