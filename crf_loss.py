import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
from math import exp
import random

#from torchvision.utils import save_image
#from torchvision import datasets

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

kernel_size = 9 #gaussian kernel dimension
dilation = 1 #cheating :) The "real" dimension of the gaussian kernel is kernel size, but the "effective" dimension is (kernel_size*dilation + 1)
padding = (kernel_size // 2) * dilation #do not touch this
bs = 4 #batch size
win = 256 #window size

sigma_X = 3.0 #for distance gaussian
sigma_I = 0.1 #for RGB/grayscale gaussian

sample_interval = 20 # sample image every

class kernel_loss(torch.nn.Module):

	def sub_kernel(self):
		filters = kernel_size * kernel_size
		middle = kernel_size // 2
		kernel = Variable(torch.zeros((filters, 1, kernel_size, kernel_size))).cuda()
		for i in range(kernel_size):
			for j in range(kernel_size):
				kernel[i*kernel_size+j, 0, i, j] = -1
				kernel[i*kernel_size+j, 0, middle, middle] = kernel[i*kernel_size+j, 0, middle, middle] + 1
		return kernel
	
	def dist_kernel(self):
		filters = kernel_size * kernel_size
		middle = kernel_size // 2
		kernel = Variable(torch.zeros((bs, filters, 1, 1))).cuda()

		for i in range(kernel_size):
			for j in range(kernel_size):
				ii = i - middle
				jj = j - middle
				distance = pow(ii,2) + pow(jj,2)
				kernel[:, i*kernel_size+j, 0, 0] = exp(-distance / pow(sigma_X,2))
		#print(kernel.view(4,1,kernel_size,kernel_size))
		return kernel
	
	def central_kernel(self):
		filters = kernel_size * kernel_size
		middle = kernel_size // 2
		kernel = Variable(torch.zeros((filters, 1, kernel_size, kernel_size))).cuda()
		for i in range(kernel_size):
			for j in range(kernel_size):
				kernel[i*kernel_size+j, 0, middle, middle] = 1
		return kernel
	
	def select_kernel(self):
		filters = kernel_size * kernel_size
		middle = kernel_size // 2
		kernel = Variable(torch.zeros((filters, 1, kernel_size, kernel_size))).cuda()
		for i in range(kernel_size):
			for j in range(kernel_size):
				kernel[i*kernel_size+j, 0, i, j] = 1
		return kernel
		
	def color_tensor(self, x):
		result = Variable(torch.zeros((bs, kernel_size*kernel_size, win-2*padding, win-2*padding))).cuda()

		for i in range(x.shape[1]):
			channel = x[:,i,:,:].unsqueeze(1)
			sub = nn.Conv2d(in_channels=1, out_channels=kernel_size*kernel_size, kernel_size=kernel_size, bias=False, padding=0, dilation=dilation)
			sub.weight.data = self.sub_matrix
			color = sub(channel)
			color = torch.pow(color,2)
			result = result + color
			
		result = torch.exp(-result / pow(sigma_I,2))
		return result

	def probability_tensor(self, y):
		conv = nn.Conv2d(in_channels=1, out_channels=kernel_size*kernel_size, kernel_size=kernel_size, bias=False, padding=0, dilation=dilation)
		conv.weight.data = self.select_matrix
		prob = conv(y)
		return prob

	#def probability_central(self, y):
	#	conv = nn.Conv2d(in_channels=1, out_channels=kernel_size*kernel_size, kernel_size=kernel_size, bias=False, padding=padding)
	#	conv.weight.data = self.one_matrix
	#	prob = conv(y)
	#	return prob

	def __init__(self):
		super(kernel_loss,self).__init__()
		#self.softmax = nn.Softmax(dim=1)
		self.dist_tensor = self.dist_kernel()
		#self.one_matrix = self.central_kernel()
		self.select_matrix = self.select_kernel()
		self.sub_matrix = self.sub_kernel() #shape: [filters, 1, h, w]

		
	def forward(self,x,y):
		"""
		x --> Image. It can also have just 1 channel (grayscale). Values between 0 and 1
		y --> Mask. Values between 0 and 1
		"""
		#y = self.softmax(y)
		y0 = y[:,0,:,:].unsqueeze(1) #build: 0, background: 1, default 1
		y1 = y[:,1,:,:].unsqueeze(1) #build: 1, background: 0, default 0
		y0p = y0[:,:,padding:-padding,padding:-padding]
		y1p = y1[:,:,padding:-padding,padding:-padding]
		
		W = self.color_tensor(x)
		W = (W * self.dist_tensor.expand_as(W))

		potts_loss_0 = y0p.expand_as(W) * W * self.probability_tensor(y1)
		potts_loss_1 = y1p.expand_as(W) * W * self.probability_tensor(y0)

		numel = potts_loss_0.numel()
		#ncut_loss_0 = (potts_loss_0 / (self.probability_tensor(y0) * W)).mean()
		#ncut_loss_1 = (potts_loss_1 / (self.probability_tensor(y1) * W)).mean()

		"""
		if random.randint(0,sample_interval) == 0:
			r = random.randint(0,20)

			img = torch.mean(W, dim=1).unsqueeze(1)
			#amin = torch.min(img)
			#amax = torch.max(img)
			#img = (img - amin) / (amax - amin)
			save_image(img, "./debug/%d_img.png" % r, nrow=2)

			#img2 = torch.mean(potts_loss_0, dim=1).unsqueeze(1)
			#amin = torch.min(img2)
			#amax = torch.max(img2)
			#img2 = (img2 - amin) / (amax - amin)
			#save_image(img2, "./debug/%d_b.png" % r, nrow=2)

			img3 = torch.mean(potts_loss_0, dim=1).unsqueeze(1)
			#amin = torch.min(img3)
			#amax = torch.max(img3)
			#img3 = (img3 - amin) / (amax - amin)
			save_image(img3, "./debug/%d_loss.png" % r, nrow=2)

			#img4 = torch.mean(loss_matrix, dim=1).unsqueeze(1)
			##amin = torch.min(img4)
			##amax = torch.max(img4)
			##img4 = (img4 - amin) / (amax - amin)
			#save_image(img4, "./debug/%d_d.png" % r, nrow=2)
			save_image(x, "./debug/%d_map.png" % r, nrow=2)
		"""
		
		potts_loss_0 = (potts_loss_0).mean()
		potts_loss_1 = (potts_loss_1).mean()
		potts_loss = potts_loss_0 + potts_loss_1

		return potts_loss

		"""
		#ncut_loss_0 = potts_loss_0 / (self.probability_tensor(y0) * W).mean()
		#ncut_loss_1 = potts_loss_1 / (self.probability_tensor(y1) * W).mean()
		ncut_loss_0 = potts_loss_0 / (y0p.expand_as(W) * W).mean()
		ncut_loss_1 = potts_loss_1 / (y1p.expand_as(W) * W).mean()

		#ncut_loss_0 = ncut_loss_0.mean()
		#ncut_loss_1 = ncut_loss_1.mean()
		ncut_loss = ncut_loss_0 + ncut_loss_1

		#potts_loss = potts_loss_0 + potts_loss_1
		#ncut_loss = ncut_loss_0 + ncut_loss_1

		return (potts_loss, ncut_loss, numel)
		"""

