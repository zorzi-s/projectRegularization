import numpy as np
import cv2
import glob
from tqdm import tqdm
import random
from skimage import io
from skimage.segmentation import mark_boundaries

import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
import numpy as np

import gdal

import variables as var


def sample_images(sample_index, img, masks):
    batch = img.shape[0]

    img = img.permute(0,2,3,1)

    for i in range(len(masks)):
        masks[i] = masks[i].permute(0,2,3,1)

    img = img.cpu().numpy()
    ip = np.uint8(img * 255)
    for i in range(len(masks)):
        masks[i] = masks[i].detach().cpu().numpy()
        masks[i] = np.argmax(masks[i], axis=-1)
        masks[i] = np.uint8(masks[i] * 255)

    line_mode = "inner"

    for i in range(len(masks)):
        row = np.copy(ip[0,:,:,:])
        line = cv2.Canny(masks[i][0,:,:], 0, 255)
        row = mark_boundaries(row, line, color=(1,1,0), mode=line_mode) * 255#, outline_color=(self.red,self.greed,0))
        for b in range(1,batch):
            pic = np.copy(ip[b,:,:,:])
            line = cv2.Canny(masks[i][b,:,:], 0, 255)
            pic = mark_boundaries(pic, line, color=(1,1,0), mode=line_mode) * 255#, outline_color=(self.red,self.greed,0))
            row = np.concatenate((row, pic), 1)
        masks[i] = row

    img = np.concatenate(masks, 0)
    img = np.uint8(img)
    io.imsave(var.DEBUG_DIR + "debug_%s.png" % str(sample_index), img)


class LossBuffer():
    def __init__(self, max_size=100):
        self.data = []
        self.max_size = max_size

    def push(self, data):
        self.data.append(data)    
        if len(self.data) > self.max_size:
            self.data = self.data[1:]
        return sum(self.data) / len(self.data)


class LambdaLR():
    def __init__(self, n_batches, decay_start_batch):
        assert ((n_batches - decay_start_batch) > 0), "Decay must start before the training session ends!"
        self.n_batches = n_batches
        self.decay_start_batch = decay_start_batch

    def step(self, batch):
        if batch > self.decay_start_batch:
            factor = 1.0 - (batch - self.decay_start_batch) / (self.n_batches - self.decay_start_batch)
            if factor > 0:
                return factor
            else:
                return 0.0
        else:
            return 1.0
