#========================================================================================#
#                                    proto net, torch                                    #
#                                       author: Yi                                       #
#                                  dataset: miniimagenet                                 #
#                        Prototypical Networks for Few-Shot Learning                     #
#                                      19, Oct 31                                        #
#                                        model.py                                        #
#========================================================================================#

#packages
import torch.nn as nn

#model
def conv_block(in_channels, out_channels):
	bn = nn.BatchNorm2d(out_channels)

	nn.init.uniform_(bn.weight)
	
	return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
		bn,
		nn.ReLU(),
		nn.MaxPool2d(2))

class ProtoNet(nn.Module):
	'''
	description:
		protonet architecture
	methods:
		__init__, forward
	'''
	def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
		super(ProtoNet, self).__init__()
		
		self.encoder = nn.Sequential(
			conv_block(x_dim, hid_dim),
			conv_block(hid_dim, hid_dim),
			conv_block(hid_dim, hid_dim),
			conv_block(hid_dim, z_dim))
		
		self.out_channels = 1600

	def forward(self, x):
		x = self.encoder(x)
		return x.view(x.size(0), -1)