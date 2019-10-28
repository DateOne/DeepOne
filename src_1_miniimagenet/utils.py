#=============================================================================================================#
#                                                DeepOne/utils                                                #
#                                                 Author: Yi                                                  #
#                                             dataset: miniimagenet                                           #
#                                               date: 19, Oct 27                                              #
#=============================================================================================================#

#packages
import os
import shutil

import pprint

import numpy as np

from torch.utils.data import DataLoader

from dataset_and_sampler import MiniImagenetBatchSampler, MiniImagenetWholeBatchSampler

#pprint
_utils_pp = pprint.PrettyPrinter()
def pprint(x):
	_utils_pp.pprint(x)

#set device
def set_device(x):
	os.environ['CUDA_VISIBLE_DEVICES'] = x
	print('using gpu: ', x)

#ensure path
def ensure_path(path):
	if os.path.exists(path):
		if input('{} exists, remove it?'.format(path)) != 'n':
			shutil.rmtree(path)
			os.makedirs(path)
	else:
		os.mkdirs(path)

#averger
def Avenger():
	'''
	ok, I know it's Averger which makes some calculation convenient. I'm not stupid.
	I just think Avenger is a cooler name.
	methods: __init__, add, item
	'''
	def __init__(self):
		self.n = 0
		self.res = 0
	
	def add(self, x):
		self.res = (self.res * self.n + x) / (self.n + 1)
		self.n += 1

	def item(self):
		return self.res

#euclidean distance
def euclidean_distance(a, b):
	n = a.shape[0]   #number of queries
	m = b.shape[0]   #training_way
	a = a.unsqueeze(1).expand(n, m, -1)
	b = b.unsqueeze(0).expand(n, m, -1)
	logits = -((a - b) ** 2).sum(dim=2)   #number of queries times training way with each element being their distance
	return logits

#complement
def complement(batch, tr_dataset, model, opt):
	'''
	fine the most related batch in training_dataset replace the validation or testing batch
	pass these training classes into the model and find the protos, find the closest protos 
	to batch protos and resample, return new batch
	'''
	batch_data, _ = batch
	batch_protos = model(batch_data).reshape(opt.shot, opt.way, -1).mean(dim=0)

	training_dataset = tr_dataset
	training_whole_sampler = MiniImagenetWholeBatchSampler(training_dataset.labels)
	training_dataloader = DataLoader(
		dataset=training_dataset,
		batch_sampler=training_whole_sampler,
		num_workers=8,
		pin_memory=True)
	for whole_tr_batch in training_dataloader:
		whole_tr_data, _ = whole_tr_batch
	whole_tr_protos = model(batch_data)
	whole_tr_protos = whole_tr_protos.reshape(
		training_whole_sampler.num_samples, training_whole_sampler.num_classes, -1).mean(dim=0)
	#get the distance between two kinds of distances
	#get the closest protos in training set and sample a new batch
	#return this new batch

#relation
def relation(grad, grad_tr):
	'''
	calculate the cross_product of training gradients and validation gradients or testing gradients
	'''
	#this shouldn't be hard
	pass

#get model
def get_model(model, memory):
	'''
	give memory change the model for every validation or testing batch
	'''
	#just add all grad and something
	pass

#memory
class Memory():
	'''
	methods: __init__, append
	description: with batch_indicator indicating the validation or testing batch,
	append the gradients to to_bug memory and to_return memory by relationships
	'''
	def __init__():
		self.batch_indicator = -1
		self.to_buy = []
		self.to_return = []

	def append(batch_indicator, grad, grad_tr):
		if self.batch_indicator != batch_indicator:
			self.to_buy.append([])
			self.to_return.append([])
			self.batch_indicator = batch_indicator

		if len(self.to_buy[self.batch_indicator]) < 100 and relation(grad, grad_tr) > max([i[1] for i in self.to_buy[self.batch_indicator]]):   #100 is just a hyper-parameter that could be changed later
			self.to_buy.pop(np.argmin(np.array([i[1] for i in self.to_buy[self.batch_indicator]])))
			self.to_buy.append([grad, relation(grad, grad_tr)])

		if len(self.to_return[self.batch_indicator]) < 100 and relation(grad, grad_tr) < min([i[1] for i in self.to_return[self.batch_indicator]]):
			self.to_return.pop(np.argmax(np.array([i[1] for i in self.to_return[self.batch_indicator]])))
			self.to_return.append([grad, relation(grad, grad_tr)])