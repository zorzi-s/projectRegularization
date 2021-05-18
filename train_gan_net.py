import os
import glob

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tqdm import tqdm
import click
import numpy as np
import cv2
from skimage.segmentation import mark_boundaries
from skimage import io
import itertools

from models import GeneratorResNet, Encoder, Discriminator
from data_loader_gan import DataLoader
from training_utils import sample_images, LossBuffer, LambdaLR
import variables as var
from crf_loss import kernel_loss



def crf_factor(batch_index, start_crf_batch, end_crf_batch, crf_initial_factor, crf_final_factor):
    if batch_index <= start_crf_batch:
        return 0.0
    elif start_crf_batch < batch_index < end_crf_batch:
        return crf_initial_factor + ((crf_final_factor - crf_initial_factor) * (batch_index - start_crf_batch) / (end_crf_batch - start_crf_batch))
    else:
        return crf_final_factor


def train(
    models_path='./saved_models_gan/', \
    restore=False, \
    batch_size=4, \
    start_batch=0, n_batches=140000, \
    start_crf_batch=60000, end_crf_batch=120000, crf_initial_factor=0.0, crf_final_factor=175.0, \
    start_lr_decay=120000, \
    start_lr=0.00004, win_size=256, sample_interval=20, backup_interval=5000):

    patch_size = int(win_size / pow(2, 4))

    Tensor = torch.cuda.FloatTensor

    e1 = Encoder(channels=3+2)
    e2 = Encoder(channels=2)
    net = GeneratorResNet()
    disc = Discriminator()

    if restore:
        print("Restoring model number %d" % start_batch)
        e1.load_state_dict(torch.load(models_path + "E%d_e1" % start_batch))
        e2.load_state_dict(torch.load(models_path + "E%d_e2" % start_batch))
        net.load_state_dict(torch.load(models_path + "E%d_net" % start_batch))
        disc.load_state_dict(torch.load(models_path + "E%d_disc" % start_batch))

    e1 = e1.cuda()
    e2 = e2.cuda()
    net = net.cuda()
    disc = disc.cuda()

    os.makedirs(models_path, exist_ok=True)
    
    loss_0_buffer = LossBuffer()
    loss_1_buffer = LossBuffer()
    loss_2_buffer = LossBuffer()
    loss_3_buffer = LossBuffer()
    loss_4_buffer = LossBuffer()
    loss_5_buffer = LossBuffer()

    gen_obj = DataLoader(bs=batch_size, nb=n_batches, ws=win_size)
    
    # Optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(net.parameters(), e1.parameters(), e2.parameters()), lr=start_lr)    
    optimizer_D = torch.optim.Adam(disc.parameters(), lr=start_lr)

    # Learning rate update schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_batches, start_lr_decay).step)
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(n_batches, start_lr_decay).step)

    bce_criterion = nn.BCELoss()
    bce_criterion = bce_criterion.cuda()

    densecrflosslayer = kernel_loss()
    densecrflosslayer = densecrflosslayer.cuda()

    loader = gen_obj.generator()
    train_iterator = tqdm(loader, total=(n_batches + 1 - start_batch))
    img_index = 0

    for batch_index, (rgb, gti, seg) in enumerate(train_iterator):

        batch_index = batch_index + start_batch

        rgb = Variable(Tensor(rgb))
        gti = Variable(Tensor(gti))
        seg = Variable(Tensor(seg))

        rgb = rgb.permute(0,3,1,2)
        gti = gti.permute(0,3,1,2)
        seg = seg.permute(0,3,1,2)

        # Adversarial ground truths
        ones = Variable(Tensor(np.ones((batch_size, 1, patch_size, patch_size))), requires_grad=False)
        zeros = Variable(Tensor(np.zeros((batch_size, 1, patch_size, patch_size))), requires_grad=False)
        valid = torch.cat((ones, zeros), dim=1)
        fake = torch.cat((zeros, ones), dim=1)

        # ------------------
        #  Train Generators
        # ------------------

        #e1.train()
        #e2.train()
        #net.train()

        optimizer_G.zero_grad()

        reg = net(e1([rgb, seg]))
        rec = net(e2([gti]))

        # Identity loss (reconstruction loss)
        loss_rec_1 = bce_criterion(reg, seg)
        loss_rec_2 = bce_criterion(rec, gti)

        # GAN loss
        loss_GAN = bce_criterion(disc(reg), valid)

        # CRF loss
        pot_multiplier = crf_factor(batch_index, start_crf_batch, end_crf_batch, crf_initial_factor, crf_final_factor)
        loss_pot = densecrflosslayer(rgb, reg)
        loss_pot = loss_pot.cuda()

        # Total loss
        loss_G = 3 * loss_GAN + 1 * loss_rec_1 + 3 * loss_rec_2 + pot_multiplier * loss_pot

        loss_G.backward()
        optimizer_G.step()


        # -----------------------
        #  Train Discriminator A
        # -----------------------

        #disc.train()

        optimizer_D.zero_grad()

        loss_real = bce_criterion(disc(rec.detach()), valid)
        loss_fake = bce_criterion(disc(reg.detach()), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Update LR
        # --------------

        lr_scheduler_G.step(batch_index)
        lr_scheduler_D.step(batch_index)

        for g in optimizer_D.param_groups:
        	current_lr = g['lr'] 

        # --------------
        #  Log Progress
        # --------------

        status = "[Batch %d][D loss: %f][G loss: %f, adv: %f, rec1: %f, rec2: %f][pot: %f, pot_mul: %f][lr: %f]" % \
        (batch_index, \
        loss_0_buffer.push(loss_D.item()), \
        loss_1_buffer.push(loss_G.item()), loss_2_buffer.push(loss_GAN.item()), loss_3_buffer.push(loss_rec_1.item()), loss_4_buffer.push(loss_rec_2.item()), 
        loss_5_buffer.push(loss_pot.item()), pot_multiplier, current_lr, )

        train_iterator.set_description(status)

        if (batch_index % sample_interval == 0):
            img_index += 1
            void_mask = torch.zeros(gti.shape).cuda()
            sample_images(img_index, rgb, [void_mask, gti, rec, seg, reg])
            if img_index >= 100:
                img_index = 0

        if (batch_index % backup_interval == 0):
            torch.save(e1.state_dict(), models_path + "E" + str(batch_index) + "_e1")
            torch.save(e2.state_dict(), models_path + "E" + str(batch_index) + "_e2")
            torch.save(net.state_dict(), models_path + "E" + str(batch_index) + "_net")
            torch.save(disc.state_dict(), models_path + "E" + str(batch_index) + "_disc")

        
if __name__ == '__main__':
    train()
