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

import torch
from torch.utils.data import DataLoader

from dataset_and_sampler import MiniImagenetBatchSampler, MiniImagenetWholeBatchSampler, FakeBatchSampler

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
		os.makedirs(path)

#averger
class Avenger():
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
def complement(batch_data, tr_dataset, model, opt, mode):
	'''
	fine the most related batch in training_dataset to replace the validation or testing batch
	pass these training classes into the model and find the protos, find the closest protos 
	to batch protos and resample, return new batch
	'''
	if mode == 'val':
		num_queries =  opt.query
	else:
		num_queries = opt.testing_query

	batch_protos = model(batch_data).reshape(opt.shot + num_queries, opt.way, -1).mean(dim=0)   #one-dimensional list of length of way with each element being the class proto
	#print(batch_protos.shape)
	training_dataset = tr_dataset
	training_whole_sampler = MiniImagenetWholeBatchSampler(training_dataset.labels, 64)   #64 is the number of classes in training set
	training_dataloader = DataLoader(
		dataset=training_dataset,
		batch_sampler=training_whole_sampler,
		num_workers=8,
		pin_memory=True)

	tr_protos = list()

	for i, tr_batch in enumerate(training_dataloader):
		tr_data, _ = [_.cuda() for _ in tr_batch]   #whole training data will be like (class1, sample1), (class2, sample1), ... (classn, sample1), (class1, sample2), ... (classn, samplen)
		tr_proto = model(tr_data).reshape(12, 1, -1).mean(dim=0)   #one-dimensional list of length of way with each element being the class proto
		tr_protos.append(tr_proto)
	#print(tr_protos)
	tr_protos = torch.cat(tr_protos, 0)
	#print(tr_protos.shape)

	batch_protos = batch_protos.unsqueeze(1).expand(opt.way, training_whole_sampler.num_classes, -1)
	tr_protos = tr_protos.expand(opt.way, training_whole_sampler.num_classes, -1)
	res = ((batch_protos - tr_protos) ** 2).sum(2)

	#print(res.shape)

	tr_idcs = torch.argmax(res, dim=1)   #one-dimensional Tensor with each element being the index of training set class to replace the batch

	#print(tr_idcs.shape)
	
	#print(tr_idcs)
	training_fake_sampler = FakeBatchSampler(
		labels=training_dataset.labels,
		class_idcs=tr_idcs,
		num_samples=opt.shot + num_queries)
	training_fake_dataloader = DataLoader(
		dataset=training_dataset,
		batch_sampler=training_fake_sampler,
		num_workers=8,
		pin_memory=True)

	for i, fake_batch in enumerate(training_fake_dataloader):
		fake_batch = fake_batch

	return fake_batch

#relation
def relation(grad, grad_tr):
	'''
	calculate the cross_product of training gradients and validation gradients or testing gradients
	oh, this is really hard
	'''
	new_grad = list()
	new_grad_tr = list()

	for module_grad in grad:
		module_grad = module_grad.reshape(-1)
		for e in module_grad:
			new_grad.append(e)

	for module_grad in grad_tr:
		module_grad =  module_grad.reshape(-1)
		for e in module_grad:
			new_grad_tr.append(e)

	grad = torch.Tensor(new_grad)
	grad_tr = torch.Tensor(new_grad_tr)

	rel = torch.dot(grad, grad_tr) / (((grad ** 2).sum() ** 0.5) * ((grad_tr ** 2).sum() ** 0.5))

	return rel

#get model
def get_model(model, memory, batch_indicator):
	'''
	give memory change the model for every validation or testing batch
	'''
	to_buy = memory.to_buy[batch_indicator]
	to_return = memory.to_return[batch_indicator]

	for i, f in enumerate(model.parameters()):
		for j in range(100):   #100 is a hyper-parameter
			f.data.add_(0.001 * to_buy[j][i])

	for i, f in enumerate(model.parameters()):
		for j in range(100):   #100 is a hyper-parameter
			f.data.sub_(0.001 * to_return[j][i])

	return model

#memory
class Memory():
	'''
	methods: __init__, append
	description: with batch_indicator indicating the validation or testing batch,
	append the gradients to to_bug memory and to_return memory by relationships
	'''
	def __init__(self):
		self.batch_indicator = -1
		self.to_buy = []
		self.to_return = []

	def append(self, batch_indicator, grad, grad_tr):   #this function could be simplified
		if self.batch_indicator != batch_indicator:
			self.to_buy.append([[0, 0]])
			self.to_return.append([0, 100])
			self.batch_indicator = batch_indicator

		if len(self.to_buy[self.batch_indicator]) < 100 and relation(grad, grad_tr) > max([i[1] for i in self.to_buy[self.batch_indicator]]):   #100 is just a hyper-parameter that could be changed later
			self.to_buy[self.batch_indicator].pop(np.argmin(np.array([i[1] for i in self.to_buy[self.batch_indicator]])))
			self.to_buy[self.batch_indicator].append([grad, relation(grad, grad_tr)])

		if len(self.to_return[self.batch_indicator]) < 100 and relation(grad, grad_tr) < min([i[1] for i in self.to_return[self.batch_indicator]]):
			self.to_return[self.batch_indicator].pop(np.argmax(np.array([i[1] for i in self.to_return[self.batch_indicator]])))
			self.to_return[self.batch_indicator].append([grad, relation(grad, grad_tr)])
