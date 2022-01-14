import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
sys.path.insert(0, '/host/src')
from resnet_dilated import Resnet34_8s
from resnet import resnet34

class Model(nn.Module):
	def __init__(self, num_keypoints, pretrained=False, channels=2, num_classes=3, img_height=480, img_width=640):
		super(Model, self).__init__()
		self.num_keypoints = num_keypoints
		self.num_outputs = self.num_keypoints
		self.img_height = img_height
		self.img_width = img_width
		self.resnet = resnet34(fully_conv=False,
                                       channels=channels,
                                       pretrained=pretrained,
                                       output_stride=8,
									   num_classes=num_classes, 
                                       remove_avg_pool_layer=False)

	def forward(self, x):
		output = self.resnet(x) 
		return output

if __name__ == '__main__':
	model = KeypointsGauss(4).cuda()
	x = torch.rand((1,3,480,640)).cuda()
	result = model.forward(x)
	print(x.shape)
	print(result.shape)
