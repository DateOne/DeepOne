#=============================================================================================================#
#                                                DeepOne/train                                                #
#                                                 Author: Yi                                                  #
#                                             dataset: miniimagenet                                           #
#                                               date: 19, Oct 23                                              #
#=============================================================================================================#

#packages
import os
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import time

from utils import pprint, set_device, ensure_path, Avenger
from utils import euclidean_distance
from dataset_and_sampler import MiniImagenet, MiniImagenetBatchSampler
from model import ProtoNet

#main
if __name__ == '__main__':
	parser = argparse.ArgumentParser('DeepOne training arguments')
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
		default=200)   #should consider this number
	parser.add_argument(
		'-tr_b', '--training_batch', type=int,
		help='number of batches',
		default=500)   #I thought it could be bigger
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
		'lr_s', '--learning_rate_step', type=int,
		help='learning rate step size',
		default=20)
	parser.add_argument(
		'-tr_w', '--training_way', type=int,
		help='number of ways for meta-tasks in training',
		default=30)
	parser.add_argument(
		'-w', '--way', type=int,
		help='number of ways for meta-tasks')
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
		help='number of query samples for meta-tasks in testing')
	parser.add_argument(
		'-d', '--device',
		help='device information',
		default='0')
	parser.add_argument(
		'-sv_r', '--save_root',
		help='save root information (not the dataset root)',
		default='save')
	args = parser.prase_args()
	pprint(vars(args))

	set_device(args.device)
	ensure_path(args.save_root)

	training_dataset = MiniImagenet('train')
	training_sampler = MiniImagenetBatchSampler(
		training_dataset.labels,
		num_batches=args.training_batch,
		num_classes=args.training_way,
		num_samples=args.shot + args.query)
	training_dataloader = DataLoader(
		dataset=training_dataset,
		batch_sampler=training_sampler,
		num_workers=8,
		pin_memory=True)
	print('***training dataset set***')

	validation_dataset = MiniImagenet('val')
	validation_sampler = MiniImagenetBatchSampler(
		validation_dataset.labels,
		num_batches=args.validation_batch,   #this parameter needs more consideration
		num_classes=args.way,
		num_samples=args.shot + args.query)
	validation_dataloader = DataLoader(
		dataset=validation_dataset,
		batch_sampler=validation_sampler,
		num_workers=8,
		pin_memory=True)
	print('***validation dataset set***')

	testing_dataset = MiniImagenet('test')
	testing_sampler = MiniImagenetBatchSampler(
		testing_dataset.labels,
		num_batches=args.testing_batch,
		num_classes=args.way,
		num_samples=args.shot + args.testing_query)
	testing_dataloader = DataLoader(
		dataset=testing_dataset,
		batch_sampler=testing_sampler,
		num_workers=8,
		pin_memory=True)
	print('***testing dataset set***')

	model = ProtoNet().cuda()
	print('***model set***')

	optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
	lr_scheduler = optim.lr_scheduler.StepLR(
		optimizer,
		gamma=args.learning_rate_gamma,
		step_size = args.learning_rate_step)
	print('***optimizer set***')

	def save_model(name):
		torch.save(model.state_dict(), os.path.join(args.save_root, name + '.pth'))

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

		for i, tr_batch in enumerate(training_dataloader):   #in this part, we want every batch of data from the dataloader to be the same
			if i > 300 and i // 3 == 0:   #important hyper-parameter, sampling rate
				tr_grad = []
				val_grad = []
				test_grad = []
				validation_important_grads = []
				testing_important_grads = []
				for j, tr_batch in enumerate(training_dataloader):   #still, same batch after
					data, _ = [_.cuda() for _ in batch]
					p = args.shot * args.training_way
					data_shot, data_query = data[:p], data[p:]
					protos = model(data_shot).reshape(args.shot, args.training_way, -1).mean(dim=0)
					label = torch.arange(args.training_way).repeat(args.query).type(torch.cuda.LongTensor)
					logits = euclidean_distance(model(data_query), protos)
					loss = F.cross_entropy(logits, label)
					loss.backward()
					params = list(model.parameters())
					grads = [params[x].grad for x in len(params)]
					tr_grad.append(grads)
				for j, val_batch in enumerate(validation_dataloader):   #same
					p = args.shot * args.way
					data_shot, data_query = data[:p], data[p:]
					protos = model(data_shot).reshape(args.shot, args.way, -1).mean(dim=0)
					label = torch.arange(args.way).repeat(args.query).type(torch.cuda.LongTensor)
					logits = euclidean_distance(model(data_query), protos)
					loss = F.cross_entropy(logits, label)
					loss.backward()
					params = list(model.parameters())
					grads = [params[x].grad for x in len(params)]
					val_grad.append(grads)
				for j, t_batch in enumerate(testing_dataloader):   #same
					p = args.shot * args.way
					data_shot, data_query = data[:p], data[p:]
					protos = model(data_shot).reshape(args.shot, args.way, -1).mean(dim=0)
					label = torch.arange(args.way).repeat(args.testing_query).type(torch.cuda.LongTensor)
					logits = euclidean_distance(model(data_query), protos)
					loss = F.cross_entropy(logits, label)
					loss.backward()
					params = list(model.parameters())
					grads = [params[x].grad for x in len(params)]
					val_grad.append(grads)
				for g in val_grad:
					for g0 in tr_grad:
						if check(g, g0):   #check is a special function
							#add this into validation_important_grads
				for g in t_grad:
					for g0 in tr_grad:
						if check(g, g0):
							#add this into test_important_grads
			data, _ = [_.cuda() for _ in batch]
			p = args.shot * args.training_way
			data_shot, data_query = data[:p], data[p:]
			protos = model(data_shot).reshape(args.shot, args.training_way, -1).mean(dim=0)
			label = torch.arange(args.training_way).repeat(args.query).type(torch.cuda.LongTensor)
			logits = euclidean_distance(model(data_query), protos)
			loss = F.cross_entropy(logits, label)
			pred = torch.argmax(logits, dim=1)
			acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()

			print('=== epoch: {}, train: {}/{}, loss={:.4f} acc={:.4f} ==='.format(epoch, i, len(training_dataloader), loss.item(), acc))

			training_loss.add(loss.item())
			training_acc.add(acc)
			
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		training_loss = training_loss.item()
		training_acc = training_acc.item()

		model.eval()   #might can't use eval and train

		validation_loss = Avenger()
		validation_acc = Avenger()

		for i, batch in enumerate(validation_dataloader, 1):
			data, _ = [_.cuda() for _ in batch]
			p = args.shot * args.way
			data_shot, data_query = data[:p], data[p:]

			model = change_model(model, grads)   #a function to change the model

			protos = model(data_shot)
			protos = protos.reshape(args.shot, args.way, -1).mean(dim=0)

			label = torch.arange(args.way).repeat(args.query)
			label = label.type(torch.cuda.LongTensor)

			logits = euclidean_distance(model(data_query), protos)
			loss = F.cross_entropy(logits, label)
			pred = torch.argmax(logits, dim=1)
			acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()

			validation_loss.add(loss.item())
			validation_acc.add(acc)

		validation_loss = validation_loss.item()
		validation_acc = validation_acc.item()

		print('=== epoch {}, val, loss={:.4f} acc={:.4f} ===\n\n'.format(epoch, validation_loss, validation_acc))

		if validation_acc > tr_log['best_acc']:
			tr_log['best_acc'] = validation_acc
			save_model('best')

		tr_log['training_loss'].append(training_loss)
		tr_log['training_acc'].append(training_acc)
		tr_log['validation_loss'].append(validation_loss)
		tr_log['validation_acc'].append(validation_acc)

		torch.save(tr_log, os.path.join(args.save_root, 'tr_log'))

		save_model('last')

		if (epoch + 1) % 20 == 0:
			save_model('epoch-{}'.format(epoch))

		time_elapsed = time.time() - since

		print('\n\n===========================\ntraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))