# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import time
#import os 
#os.environ['CUDA_ENABLE_DEVICES'] = '0' 
import torch
import torch.nn.functional as F
import util
torch.cuda.empty_cache()

def finetune_centroids(train_loader, student, teacher, criterion, optimizer, epoch=0, n_iter=-1, verbose=False):
    """
    Student/teacher distillation training loop.

    Remarks:
        - The student has to be in train() mode as this function will not
          automatically switch to it for finetuning purposes
    """

    #student.train()
    losses = util.AverageMeter()
  
    for i, (input, target) in enumerate(train_loader):
        # early stop
        if i >= n_iter: break

        
        print('Epoch {}starts'.format(i))
    
        # cuda
        input = input.cuda()
        
        teacher_, _t=teacher(input)
        student_, _s=student(input)
        student_probs = F.softmax(student_, dim=1) 
        teacher_probs = F.softmax(teacher_, dim=1)
        loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')
        losses.update(loss.item(), input.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bpd=util.bits_per_dim(input, losses.avg)
        


    return losses.avg


def evaluate(test_loader, model, criterion, n_iter=-1, verbose=False, device='cuda'):
    """
    Standard evaluation loop.
    """
    loss_meter = util.AverageMeter()

    # switch to evaluate mode
    model.train()

    with torch.no_grad():
        #end = time.time()
        #bpd = 0
        for i, (x, target) in enumerate(test_loader):
            # early stop
            if i >=100: break

            x = x.to('cuda')
            
            z, sldj = model(x, reverse=False)
            loss = criterion(z, sldj)
            loss_meter.update(loss.item(), x.size(0))
            bpd=util.bits_per_dim(x, loss_meter.avg)
            
            
        return bpd


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """
    Pretty and compact metric printer.
    """

    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    """

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
