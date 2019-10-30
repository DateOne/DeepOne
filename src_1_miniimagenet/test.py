#=============================================================================================================#
#                                                DeepOne/test                                                 #
#                                                 Author: Yi                                                  #
#                                             dataset: miniimagenet                                           #
#                                               date: 19, Oct 27                                              #
#=============================================================================================================#

#packages
import argparse

import torch
from torch.utils.data import DataLoader

from utils import pprint, set_device, Avenger
from utils import euclidean_distance, get_model
from dataset_and_sampler import MiniImagenet, MiniImagenetBatchSampler
from model import ProtoNet
from train import t_memory   #is that right?

#main
if __name__ == '__main__':
	parser = argparse.ArgumentParser('DeepOne testing arguments')
	parser.add_argument(
		'-t_b', '--testing_batch', type=int,
		help='number of batchs in testing',
		default=50)
	parser.add_argument(
		'-w', '--way', type=int,
		help='number of ways for meta-tasks',
		default=5)
	parser.add_argument(
		'-s', '--shot', type=int,
		help='number of shots for meta-tasks',
		default=1)
	parser.add_argument(
		'-t_q', '--testing_query', type=int,
		help='number of query samples for meta-tasks in testing',
		default=30)
	parser.add_argument(
		'-d', '--device',
		help='device information',
		default='0')
	args.parser.parse_args()
	pprint(vars(args))

	MODEL_ROOT = 'save/best.pth'
	set_device(args.device)

	dataset = MiniImagenet('test')
	sampler = MiniImagenetBatchSampler(
		dataset.labels,
		num_batches=args.iterations,
		num_classes=args.way,
		num_samples=args.shot + args.testing_query)
	dataloader = DataLoader(
		dataset,
		batch_sampler=sampler,
		num_workers=8,
		pin_memory=True)

	model = ProtoNet().cuda()
	model.load_state_dict(torch.load(MODEL_ROOT))

	model.eval()

	test_acc = Avenger()

	for i, t_batch in enumerate(dataloader):
		data, _ = [_.cuda() for _ in batch]
		p = args.way * args.shot
		data_shot, data_query = data[:p], data[p:]

		t_model = model
		t_model = get_model(t_model, t_memory, i)

		protos = t_model(data_shot).reshape(args.shot, args.way, -1).mean(dim=0)
		label = torch.arange(args.way).repeat(args.testing_query).type(torch.cuda.LongTensor)

		logits = euclidean_distance(t_model(data_query), protos)
		pred = torch.argmax(logits, dim=1)
		acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()

		test_acc.add(acc)

		print('=== batch {}: {:.2f}({:.2f})'.format(i, test_acc.item() * 100, acc * 100))
