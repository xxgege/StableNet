import os
import random
import shutil
import time

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.autograd import Variable
from utilis.matrix import accuracy
from utilis.meters import AverageMeter, ProgressMeter

from training.reweighting import weight_learner


def train(train_loader, model, criterion, optimizer, epoch, args, tensor_writer=None):
    ''' TODO write a dict to save previous featrues  check vqvae,
        the size of each feature is 512, os we need a tensor of 1024 * 512
        replace the last one every time
        and a weight with size of 1024,
        replace the last one every time
        TODO init the tensors
    '''

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        output, cfeatures = model(images)
        pre_features = model.pre_features
        pre_weight1 = model.pre_weight1

        if epoch >= args.epochp:
            weight1, pre_features, pre_weight1 = weight_learner(cfeatures, pre_features, pre_weight1, args, epoch, i)

        else:
            weight1 = Variable(torch.ones(cfeatures.size()[0], 1).cuda())

        model.pre_features.data.copy_(pre_features)
        model.pre_weight1.data.copy_(pre_weight1)

        loss = criterion(output, target).view(1, -1).mm(weight1).view(1)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        method_name = args.log_path.split('/')[-2]
        if i % args.print_freq == 0:
            progress.display(i, method_name)
            progress.write_log(i, args.log_path)

    tensor_writer.add_scalar('loss/train', losses.avg, epoch)
    tensor_writer.add_scalar('ACC@1/train', top1.avg, epoch)
    tensor_writer.add_scalar('ACC@5/train', top5.avg, epoch)
