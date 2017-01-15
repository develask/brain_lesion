import os
import numpy as np
import nibabel as nib
import image_functions as imf
import random as rdm

class Examples():
	def __init__(self):
		self.pairs = []
		self.images = {}
		self.results = []

	def __iter__(self):
		for el in self.pairs:
			yield el

	def __getitem__(self, num):
		return self.pairs[num]

	def remove_elem(self,ind):
		self.pairs.pop(ind) #removing its index

	def shuffle_exs(self):
		rdm.shuffle(self.pairs)

	def reset_exs(self):
		del self.pairs
		self.pairs = []

	def initilize(self, img = None):
		for f in os.walk("../data/mask/normalized/"):
			direcs = f
			break
		images_out = direcs[2]

		for i in range(len(images_out)):
			try:
				if (img == None or images_out[i].index(img)>-1):
					print("Loading image:", images_out[i])
					self.results.append(imf.OurImage(nib.load("../data/mask/normalized/" + images_out[i]).get_data(), images_out[i][0:6]))
			except:
				pass
	
	def get_examples(self,step = 1,output_type="regression"):

		# this function loads both the inputs (the subsamples) and the
		# desired outputs (the mask)
		brain = imf.OurImage(nib.load("../data/standars/MNI152_T1_1mm_first_brain_mask.nii.gz").get_data(), "brain")

		self.reset_exs()

		for im in self.results:
			print("Getting slices from:", im.name)


			slices = im.filterByImage(brain, step)

			for j in range(len(slices)):
				if output_type == "regression":
					val = im[slices[j].x][slices[j].y][slices[j].z]
					if val > 0:
						val = val/2 + 0.5
					self.pairs.append((slices[j], ######## akaso solo slices[j],
													   # para pasar info adicional a la NN
									   (1-val, val)))
				else:
					out = im[slices[j].x][slices[j].y][slices[j].z]
					klasea = ()
					if out > 0:
						#klasea = 1
						klasea = (0,1)
					else:
						klasea = (1,0)
						#klasea = 0
					self.pairs.append((slices[j], ######## akaso solo slices[j],
													   # para pasar info adicional a la NN
									   klasea))

	def valance(self, portion):
		# portion --> neg/positive rate
		print("Valancing data: (",portion,"negatives for 1 positive )")
		num_pos = 0
		for pair in self.pairs:
			if pair[1][1] > 0:
				num_pos += 1

		r = list(range(0,len(self.pairs)))
		rdm.shuffle(r)

		num_pos = num_pos * portion
		num_neg = 0
		to_be_removed = [] # list of index that should be removed
		for i in r:
			if self.pairs[i][1][1] == 0:
				if num_neg < num_pos:
					num_neg += 1
				else:
					to_be_removed.append(i)

		to_be_removed = sorted(to_be_removed,reverse=True)
		for i in to_be_removed:
			self.remove_elem(i)

	def split(self, portion):
		print("Spliting data for train test:",portion)
		self.shuffle_exs() ## shuffle the training - test examples

		tot = portion*len(self.pairs)
		i = 0

		X_train = []
		y_train = []

		X_test = []
		y_test = []


		for pair in self.pairs:
			if i<tot:
				X_train.append(pair[0])
				y_train.append(pair[1])
				i += 1
			else:
				X_test.append(pair[0])
				y_test.append(pair[1])
		return [(X_train,y_train),(X_test,y_test)]

	def load(self, img_type, crbs):
		#print("\tChecking", img_type, "type...")
		for f in os.walk("../data/"+img_type+"/normalized/"):
			direcs = f
			break
		images_in = direcs[2]
		newImages = []
		for i in range(len(images_in)):
			name = images_in[i]
			if (crbs!=None):
				for c in crbs:
					try:
						if (not img_type+"-"+name[0:6] in self.images or name.index(c)>-1):
							self.images[img_type+"-"+name[0:6]] = imf.OurImage(nib.load("../data/"+img_type+"/normalized/" + name).get_data(), name[0:6])
					except:
						pass
			else:
				if (not img_type+"-"+name[0:6] in self.images):
					self.images[img_type+"-"+name[0:6]] = imf.OurImage(nib.load("../data/"+img_type+"/normalized/" + name).get_data(), name[0:6])

	def getData(self, indexes, img_types, sample_type, dim, crbs=None):
		# img_types, e.g. FA, MO,...
		# sample_type, e.g. 2dx, 3d...
		# dim dimension of the subsample
		#print("Getting data for: ", img_types, "and", sample_type)
		for img_type in img_types:
			self.load(img_type, crbs)

		newList = []
		for el in indexes:
			data = []
			for img_type in img_types:
				data.append(self.images[img_type+"-"+el.fromIm].slice_matrix(el.x, el.y, el.z, dim, sample_type))
			newList.append(data)
		return newList


