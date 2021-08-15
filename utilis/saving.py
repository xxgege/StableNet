import os
import shutil

import torch


def save_checkpoint(state, is_best, log_path, epoch=0, filename='checkpoint.pth.tar'):
    savename = os.path.join(os.path.dirname(log_path), "epoch_" + str(epoch) + "_" + filename)
    torch.save(state, savename)
    if is_best:
        shutil.copyfile(savename, 'model_best.pth.tar')
