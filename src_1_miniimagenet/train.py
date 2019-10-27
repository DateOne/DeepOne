#=============================================================================================================#
#                                                DeepOne/train                                                #
#                                                 Author: Yi                                                  #
#                                             dataset: miniimagenet                                           #
#                                               date: 19, Oct 27                                              #
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
from utils import euclidean_distance, complement, Memory, get_model
from dataset_and_sampler import MiniImagenet, MiniImagenetBatchSampler
from model import ProtoNet

#dataloader
def init_dataset(mode):
	return MiniImagenet(mode)

def init_dataloader(mode):
	dataset = MiniImagenet(mode)
	if mode == 'train':
		num_batches = args.training_batch
		num_classes = args.training_way
		num_samples = args.shot + args.query
	elif mode == 'val':
		num_batches = args.validation_batch
		num_classes = args.way
		num_samples = args.shot + args.query
	else mode == 'test':
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
	print('*** {} set ready ***'.format(mode))
	return dataloader

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
		default=200)
	parser.add_argument(
		'-tr_b', '--training_batch', type=int,
		help='number of batches',
		default=500)
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

	set_device(args.device)
	ensure_path(args.save_root)

	training_dataset = init_dataset('train')

	training_dataloader = init_dataloader('train')
	validation_dataloader = init_dataloader('val')
	testing_dataloader = init_dataloader('test')

	model = ProtoNet().cuda()

	optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
	lr_scheduler = optim.lr_scheduler.StepLR(
		optimizer,
		gamma=args.learning_rate_gamma,
		step_size = args.learning_rate_step)

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

	t_memory = Memory()   #need some thoughts

	for epoch in range(args.epoch):
		lr.scheduler.step()

		model.train()

		training_loss = Avenger()
		training_acc = Avenger()

		val_memory = Memory()   #need some thoughts

		for i, tr_batch in enumerate(training_dataloader, 1):
			data, _ = [_.cuda for _ in batch]
			
			p = args.shot * args.training_way
			data_shot, data_query = data[:p], data[p:]

			protos = model(data_shot).reshape(args.shot, args.training_way, -1).mean(dim=0)
			
			label = torch.arange(arga.training_way).repeat(args.query).type(torch.cuda.LongTensor)
			logits = euclidean_distance(model(data_query), protos)
			loss = F.cross_entropy(logits, label)
			pred = torch.argmax(logits, dim=1)
			acc = (pred = label).type(torch.cuda.FloatTensor).mean().item()

			print('=== epoch: {}, train: {}/{}, loss={:.4f} acc={:.4f} ==='.format(epoch, i, len(training_dataloader), loss.item(), acc))

			training_loss.add(loss.item())
			training_acc.add(acc)
			
			optimizer.zero_grad()
			loss.backward()

			tr_params = list(model.parameters())
			tr_grads = [tr_params[x].grad for x in len(tr_params)]

			optimizer.step()

			for j, val_batch in enumerate(validation_dataloader, 1):
				val_batch = complement(val_batch, training_dataset, model, args)

				data, _ = [_.cuda() for _ in batch]
				p = args.shot + args.way
				data_shot, data_query = data[:p], data[p:]

				protos = model(data_shot).reshape(args.shot, args.way, -1).mean(dim=0)
				
				label = torch.arange(args.way).repeat(args.query).type(torch.cuda.LongTensor)
				logits = euclidean_distance(model(data_query), protos)
				loss = F.cross_entropy(logits, label)
				loss.backward()
				
				val_params = list(model.parameters())
				val_grads = [params[x].grad for x in len(params)]

				val_memory.append(val_grads)

			for j, t_batch in enumerate(testing_dataloader, 1):
				t_batch = complement(t_batch, training_dataset, model, args)

				data, _ = [_.cuda() for _ in batch]
				p = args.shot * args.way
				data_shot, data_query = data[:p], data[p:]
				
				protos = model(data_shot).reshape(args.shot, args.way, -1).mean(dim=0)
				label = torch.arange(args.way).repeat(args.testing_query).type(torch.cuda.LongTensor)
				logits = euclidean_distance(model(data_query), protos)
				loss = F.cross_entropy(logits, label)
				loss.backward()
				
				t_params = list(model.parameters())
				t_grads = [params[x].grad for x in len(params)]

				t_memory.append(t_grads)

		training_loss = training_loss.item()
		training_acc = training_acc.item()   #maybe the training process could be changed, you know for training acc we can even get a new model

		model.eval()

		validation_loss = Avenger()
		validation_acc = Avenger()

		for i, batch in enumerate(validation_dataloader):
			data, _ = [_.cuda() for _ in batch]
			p = args.shot * args.way
			data_shot, data_query = data[:p], data[p:]

			val_model = model
			val_model = get_model(val_model, val_memory)

			protos = val_model(data_shot).reshape(args.shot, args.way, -1).mean(dim=0)
			label = torch.arange(args.way).repeat(args.query).type(torch.cuda.LongTensor)
			logits = euclidean_distance(val_model(data_query), protos)
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