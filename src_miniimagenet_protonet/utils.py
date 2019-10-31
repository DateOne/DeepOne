#========================================================================================#
#                                    proto net, torch                                    #
#                                       author: Yi                                       #
#                                  dataset: miniimagenet                                 #
#                        Prototypical Networks for Few-Shot Learning                     #
#                                      19, Oct 31                                        #
#                                       utils.py                                         #
#========================================================================================#

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
		if input('{} exists, remove it? ([y]/n)'.format(path)) != 'n':
			shutil.rmtree(path)
			os.makedirs(path)
	else:
		os.makedirs(path)

#averager
class Avenger():
	'''
	avengers assemble!
	methods:
		__init__
		add
		item
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

#save model
def save_model(model, name, opt):
	torch.save(model.state_dict(), os.path.join(opt.save_root, 'models', name + '.pth'))

#save grad
def save_grad(model, name, opt):
	params = list(model.parameters())
	grads = [tr_params[x].grad for x in range(len(tr_params))]
	torch.save(grads, os.path.join(opt.save_root, 'grads', name))

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
		fake_data, _ = [_.cuda() for _ in fake_batch]

	return fake_data

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