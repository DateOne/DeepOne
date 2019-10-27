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
	methods:
		__init__, add, item
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
	n = a.shape[0]
	m = n.shape[0]
	a = a.unsqueeze(1).expand(n, m, -1)
	b = b.unsqueeze(0).expand(n, m, -1)
	logits = -((a - b) ** 2).sum(dim=2)
	return logits

#complement
def complement(batch, tr_dataset, model, opt):
	'''
	fine the most related batch in training_dataset replace the validation or testing batch
	pass these training classes into the model and find the protos, find the closest protos 
	to batch protos and resample, return new batch
	'''
	data, _ = batch
	batch_protos = model(data).reshape(args.shot, args.way, -1).mean(dim=0)
	#ok, might later
	# pass

#memory
class Memory():
	'''
	validation memory and testing memory
	define the relationship and how to select this fucker, very hard and very important
	'''
	pass

#get model
def get_model(model, memory):
	'''
	given memory change the model for every validation or testing batch
	'''
	pass