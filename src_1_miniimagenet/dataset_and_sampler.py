#=============================================================================================================#
#                                         DeepOne/dataset_and_sampler                                         #
#                                                 Author: Yi                                                  #
#                                             dataset: miniimagenet                                           #
#                                               date: 19, Oct 27                                              #
#=============================================================================================================#

#packages
import os
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dateset
from torchvision import transforms

#hyper-parameters
root = '../../datasets/miniimagenet'

#dataset
class MiniImagenet(Dateset):
	'''
	miniimagenet dataset
		methods: __init__, __getitem__, __len__
		description: split dataset, get image paths and labels, define transforms,
		get image and label pair, get length
	'''
	def __init__(self, mode):
		csv_path = os.path.join(root, mode + '.csv')
		lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]   #one dimensional list with each element being data path and label name

		self.data_paths = []   #to store data paths
		self.labels = []   #to store labels (every sample)

		self.label_names = []   #to store name of labels (with the length of number of classes, with character)
		label_indicator = -1

		for line in lines:
			data_path, label_name = line.split(',')
			data_path = os.path.join(root, 'images', data_path)

			if label_name not in self.label_names:
				self.label_names.append(label_name)
				label_indicator += 1

			self.data_paths.append(data_path)
			self.labels.append(label_indicator)   #one-by-one correspondence, we can see the same class cluster together

		self.transform = transforms.Compose([
			transforms.Resize(84),
			transforms.CenterCrop(84),
			transforms.ToTensor(),
			transforms.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225])])

	def __getitem__(self, idx):
		path, label = self.data_paths[idx], self.labels[idx]
		image = self.transform(Image.open(path)).convert('RGB')
		return image, label

	def __len__(self):
		return len(self.labels)

#sampler
class MiniImagenetBatchSampler():   #every batch should be the same!!!
	'''
	miniimagenet batch sampler
		methods: __init__, __iter__, __len__
		description: randomly choose some classes and randomly choose some samples from these classes
	'''
	def __init__(self, labels, num_batches, num_classes, num_samples):
		self.num_batches = num_batches
		self.num_classes = num_classes
		self.num_samples = num_samples

		labels = np.array(labels)   #labels was one-dimensional list

		self.class_class = []
		for i in range(max(labels) + 1):
			class_i = np.argwhere(labels == i).reshape(-1)
			class_i = torch.from_numpy(class_i)   #class_i is just one-dimensional array of indices
			self.class_class.append(class_i)   #class_class is two-dimensional list with each sub-list being a whole class of samples

	def __iter__(self):
		for b in range(self.num_batches):
			batch = []
			classes = torch.randperm(len(self.class_class))[:self.num_classes]
			for c in classes:
				the_class = self.class_class[c]
				samples_in_class = torch.randperm(len(the_class))[:self.num_samples]
				batch.append(the_class[samples_in_class])
			batch = torch.stack(batch).t().reshape(-1)   #it would be like: (class1, sample1), (class2, sample1), ... (classn, sample1), (class1, sample2), ... (classn, samplen)
			yield batch

	def __len__(self):
		return self.num_batches

#sampler for complement function
class MiniImagenetWholeBatchSampler():
	'''
	miniimagenet batch sampler for complement function
		methods: __init__, __iter__, __len__
		description: choose all classes and all samples
	'''
	def __init__(self, labels):
		labels = np.array(labels)
		self.class_class = []
		for i in range(max(labels) + 1):
			class_i = np.argwhere(labels == i).reshape(-1)
			class_i = torch.from_numpy(class_i)
			self.class_class.append(class_i)

		self.num_classes = max(labels) + 1
		self.num_samples = len(class_class[0])

	def __iter__(self):
		batch = torch.stack(batch).t().reshape(-1)
		yield batch

	def __len__(self):
		return 1