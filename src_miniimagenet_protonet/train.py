#========================================================================================#
#                                    proto net, torch                                    #
#                                       author: Yi                                       #
#                                  dataset: miniimagenet                                 #
#                        Prototypical Networks for Few-Shot Learning                     #
#                                      19, Oct 31                                        #
#                                       train.py                                         #
#========================================================================================#

#packages
import os
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import time

from dataset_and_sampler import MiniImagenet, MiniImagenetBatchSampler
from model import ProtoNet
from utils import pprint, set_device, ensure_path, Avenger
from utils import euclidean_distance, save_grad

#main
if __name__ == '__main__':
	
	parser = argparse.ArgumentParser('DeepOne training arguments')   #these parameters need some big change!!!
	'''
	arguments:
		number of epochs
		number of batches
		learning rate
		learning rate scheduler gamma
		learning rate scheduler step size
		number of ways for meta-tasks in training
		number of shots for meta-tasks in training
		number of query samples for meta-tasks in training
		number of ways for meta-tasks in validation
		device information
		save root information (not the dataset root) 
	'''
	parser.add_argument(
		'-e', '--epoch', type=int,
		help='number of epochs',
		default=200)
	parser.add_argument(
		'-tr_b', '--training_batch', type=int,
		help='number of batches',
		default=100)
	parser.add_argument(
		'-val_b', '--validation_batch', type=int,
		help='number of validation batches',
		default=400)
	parser.add_argument(
		'-t_b', '--testing_batch', type=int,
		help='number of batchs in testing',
		default=2000)
	parser.add_argument(
		'-lr', '--learning_rate', type=int,
		help='learning rate',
		default=0.001)
	parser.add_argument(
		'-lr_g', '--learning_rate_gamma', type=float,
		help='learning rate gamma',
		default=0.5)
	parser.add_argument(
		'-lr_s', '--learning_rate_step', type=int,
		help='learning rate step size',
		default=20)
	parser.add_argument(
		'-tr_w', '--training_way', type=int,
		help='number of ways for meta-tasks in training',
		default=30)
	parser.add_argument(
		'-w', '--way', type=int,
		help='number of ways for meta-tasks',
		default=5)
	parser.add_argument(
		'-s', '--shot', type=int,
		help='number of shots for meta-tasks',
		default=1)
	parser.add_argument(
		'-q', '--query', type=int,
		help='number of query samples for meta-tasks',
		default=15)
	parser.add_argument(
		'-t_q', '--testing_query', type=int,
		help='number of query samples for meta-tasks in testing',
		default=30)
	parser.add_argument(
		'-d', '--device',
		help='device information',
		default='0')
	parser.add_argument(
		'-sv_r', '--save_root',
		help='save root information (not the dataset root)',
		default='save')
	parser.add_argument(
		'-sd', '--manual_seed',
		help='manual seed to guarantee each batch in validation sampler and testing sampler will be same',
		default=111)
	
	args = parser.parse_args()

	pprint(vars(args))

	#dataloader
	def init_dataset(mode):
		'''
		description:
			initiate miniimagenet dataset
		'''
		return MiniImagenet(mode)

	def init_dataloader(mode):
		'''
		description:
			initiate dataloaders
		'''
		dataset = init_dataset(mode)

		if mode == 'train':
			num_batches = args.training_batch
			num_classes = args.training_way
			num_samples = args.shot + args.query

		elif mode == 'val':
			num_batches = args.validation_batch
			num_classes = args.way
			num_samples = args.shot + args.query

		else:
			num_batches = args.testing_batch
			num_classes = args.way
			num_samples = args.shot + args.testing_query

		batch_sampler = MiniImagenetBatchSampler(
			dataset.labels,
			num_batches=num_batches,
			num_classes=num_classes,
			num_samples=num_samples)

		dataloader = DataLoader(
			dataset=dataset,
			batch_sampler=batch_sampler,
			num_workers=8,
			pin_memory=True)

		print('{} set ready'.format(mode))
		
		return dataloader

	set_device(args.device)
	ensure_path(args.save_root)

	training_dataloader = init_dataloader('train')
	validation_dataloader = init_dataloader('val')
	
	model = ProtoNet().cuda()

	optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
	lr_scheduler = optim.lr_scheduler.StepLR(
		optimizer,
		gamma=args.learning_rate_gamma,
		step_size = args.learning_rate_step)

	def save_model(name):
		torch.save(model.state_dict(), os.path.join(args.save_root, 'models', name + '.pth'))

	tr_log = {}
	tr_log['args'] = vars(args)
	tr_log['training_loss'] = []
	tr_log['training_acc'] = []
	tr_log['validation_loss'] = []
	tr_log['validation_acc'] = []
	tr_log['best_acc'] = 0

	since = time.time()

	for epoch in range(args.epoch):
		lr_scheduler.step()

		model.train()

		training_loss = Avenger()
		training_acc = Avenger()

		for i, batch in enumerate(training_dataloader):
			data, _ = [_.cuda() for _ in batch]   #data will be like (class1, sample1), (class2, sample1), ... (classn, sample1), (class1, sample2), ... (classn, samplen)
			p = args.shot * args.training_way
			data_shot, data_query = data[:p], data[p:]

			protos = model(data_shot)
			protos = protos.reshape(args.shot, args.training_way, -1).mean(dim=0)   #think of that as a length-training_way Tensor with each element being the proto of its class

			logits = euclidean_distance(model(data_query), protos)   #two-dimensional Tensor (number of queries times training way) with each element being distance, the arangement of queries see data

			label = torch.arange(args.training_way).repeat(args.query)
			label = label.type(torch.cuda.LongTensor)

			loss = F.cross_entropy(logits, label)   #that makes sense
			pred = torch.argmax(logits, dim=1)
			acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()

			print('=== epoch: {}, train: {}/{}, loss={:.4f} acc={:.4f} ==='.format(epoch + 1, i + 1, len(training_dataloader), loss.item(), acc))

			training_loss.add(loss.item())
			training_acc.add(acc)
			
			optimizer.zero_grad()
			loss.backward()

			if np.random.rand() < 0.1:   #around 2000 samples saved
				save_model('epoch_{}_batch_{}'.format(epoch, i))
				save_grad('epoch_{}_batch_{}'.format(epoch, i))

			optimizer.step()

		training_loss = training_loss.item()
		training_acc = training_acc.item()

		model.eval()

		validation_loss = Avenger()
		validation_acc = Avenger()

		for i, batch in enumerate(validation_dataloader):
			data, _ = [_.cuda() for _ in batch]
			p = args.shot * args.way
			data_shot, data_query = data[:p], data[p:]

			protos = model(data_shot)
			protos = protos.reshape(args.shot, args.way, -1).mean(dim=0)

			logits = euclidean_distance(model(data_query), protos)

			label = torch.arange(args.way).repeat(args.query)
			label = label.type(torch.cuda.LongTensor)

			loss = F.cross_entropy(logits, label)
			pred = torch.argmax(logits, dim=1)
			acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()

			validation_loss.add(loss.item())
			validation_acc.add(acc)

		validation_loss = validation_loss.item()
		validation_acc = validation_acc.item()

		print('=== epoch {}, val, loss={:.4f} acc={:.4f} ==='.format(epoch + 1, validation_loss, validation_acc))

		if validation_acc > tr_log['best_acc']:
			tr_log['best_acc'] = validation_acc
			save_model('best')

		tr_log['training_loss'].append(training_loss)
		tr_log['training_acc'].append(training_acc)
		tr_log['validation_loss'].append(validation_loss)
		tr_log['validation_acc'].append(validation_acc)

		torch.save(tr_log, os.path.join(args.save_root, 'tr_log'))

		save_model('last')

		#if (epoch + 1) % 20 == 0:
		#	save_model('epoch-{}'.format(epoch))

		time_elapsed = time.time() - since
		print('training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))