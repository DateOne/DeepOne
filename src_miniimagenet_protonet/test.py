#========================================================================================#
#                                    proto net, torch                                    #
#                                       author: Yi                                       #
#                                  dataset: miniimagenet                                 #
#                        Prototypical Networks for Few-Shot Learning                     #
#                                      19, Oct 31                                        #
#                                        test.py                                         #
#========================================================================================#

#packages
import os
import argparse

import torch
import torch.nn.Functional as F
from torch.utils.data import DataLoader

from utils import pprint, set_device, Avenger
from utils import euclidean_distance, relation, complement, 
from dataset_and_sampler import MiniImagenet, MiniImagenetBatchSampler
from model import ProtoNet

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
	
	args = parser.parse_args()
	
	pprint(vars(args))

	MODEL_ROOT = 'save/best.pth'
	set_device(args.device)

	dataset = MiniImagenet('test')
	sampler = MiniImagenetBatchSampler(
		dataset.labels,
		num_batches=args.testing_batch,
		num_classes=args.way,
		num_samples=args.shot + args.testing_query)
	dataloader = DataLoader(
		dataset,
		batch_sampler=sampler,
		num_workers=8,
		pin_memory=True)

	training_dataset = MiniImagenet('train')

	model = ProtoNet().cuda()
	model.load_state_dict(torch.load(MODEL_ROOT))

	model.eval()

	test_acc = Avenger()

	for i, batch in enumerate(dataloader):
		i_model = model

		data, _ = [_.cuda() for _ in batch]

		p = args.way * args.shot
		data_shot, data_query = data[:p], data[p:]

		#get a new_model
		#walk in models and assign this model
			#get corresponding grads
			#calculate the grads on this fake data
			#calculate the relationship
			#update the model (not new_model) by this relationship 
		
		fake_data = complement(data, training_dataset, i_model, args, 'test')
		fake_data_shot, fake_data_query = fake_data[:p], fake_data[p:]

		label = torch.arange(args.way).repeat(args.testing_query)
		label = label.type(torch.cuda.LongTensor)

		temp_model = ProtoNet().cuda()

		for filename in os.listdir('save/model'):
			temp_model.load_state_dict(torch.load(os.path.join('save/models', filename)))

			fake_protos = temp_model(fake_data_shot),reshape(args.shot, args.way, -1).mean(dim=0)
			fake_logits = euclidean_distance(temp_model(fake_data_query), fake_protos)

			loss = F.cross_entropy(fake_logits, label)
			loss.backward()

			params = list(temp_model.parameters())
			grads = [params[x].grad for x in range(len(params))]

			filename_grad = filename.replace('.pth', '')
			grads_tr = torch.load(os.path.join('save/grads', filename_grad))

			rel = relation(grads, grads_tr)

			for i, f in enumerate(i_model.parameters()):
				f.data.add_(0.001 * rel * grads_tr[i])   #very important step

		protos = i_model(data_shot)
		protos = protos.reshape(args.shot, args.way, -1).mean(dim=0)

		logits = euclidean_distance(i_model(data_query), protos)
	
		pred = torch.argmax(logits, dim=1)
		
		acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
		
		test_acc.add(acc)
	
	print('=== batch {}: {:.2f}({:.2f}) ==='.format(i, test_acc.item() * 100, acc * 100))