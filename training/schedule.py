import math


def lr_setter(optimizer, epoch, args, bl=False):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    lr = args.lr
    if bl:
        lr = args.lrbl * (0.1 ** (epoch // (args.epochb * 0.5)))
    else:
        if args.cos:
            lr *= ((0.01 + math.cos(0.5 * (math.pi * epoch / args.epochs))) / 1.01)
        else:
            if epoch >= args.epochs_decay[0]:
                lr *= 0.1
            if epoch >= args.epochs_decay[1]:
                lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
