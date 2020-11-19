from typing import Optional
from torch.optim.optimizer import Optimizer
import sys
import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np


class StepwiseLR:
    """
    A lr_scheduler that update learning rate using the following schedule:

    .. math::
        \text{lr} = \text{init_lr} \times \text{lr_mult} \times (1+\gamma i)^{-p},

    where `i` is the iteration steps.

    Parameters:
        - **optimizer**: Optimizer
        - **init_lr** (float, optional): initial learning rate. Default: 0.01
        - **gamma** (float, optional): :math:`\gamma`. Default: 0.001
        - **decay_rate** (float, optional): :math:`p` . Default: 0.75
    """

    def __init__(self, optimizer: Optimizer, init_lr: Optional[float] = 0.01,
                 gamma: Optional[float] = 0.001, decay_rate: Optional[float] = 0.75):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0

    def get_lr(self) -> float:
        lr = self.init_lr * (1 + self.gamma * self.iter_num) ** (-self.decay_rate)
        return lr

    def step(self):
        """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if 'lr_mult' not in param_group:
                param_group['lr_mult'] = 1.
            param_group['lr'] = lr * param_group['lr_mult']

        self.iter_num += 1


class AverageMeter(object):
    """Computes and stores the average and current value"""

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
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ForeverDataIterator:
    """A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)


class ResizeImage(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            output size will be (size, size)
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


class AccuracyCounter:

    def __init__(self, length):
        self.Ncorrect = np.zeros(length)
        self.Ntotal = np.zeros(length)
        self.length = length

    def add_correct(self, index, amount=1):
        self.Ncorrect[index] += amount

    def add_total(self, index, amount=1):
        self.Ntotal[index] += amount

    def clear_zero(self):
        i = np.where(self.Ntotal == 0)
        self.Ncorrect = np.delete(self.Ncorrect, i)
        self.Ntotal = np.delete(self.Ntotal, i)

    def each_accuracy(self):
        self.clear_zero()
        return self.Ncorrect / self.Ntotal

    def mean_accuracy(self):
        self.clear_zero()
        return np.mean(self.Ncorrect / self.Ntotal)

    def h_score(self):
        self.clear_zero()
        common_acc = np.mean(self.Ncorrect[0:-1] / self.Ntotal[0:-1])
        open_acc = self.Ncorrect[-1] / self.Ntotal[-1]
        return 2 * common_acc * open_acc / (common_acc + open_acc)


def get_consistency(y_1, y_2, y_3, y_4, y_5):
    y_1 = torch.unsqueeze(y_1, 1)
    y_2 = torch.unsqueeze(y_2, 1)
    y_3 = torch.unsqueeze(y_3, 1)
    y_4 = torch.unsqueeze(y_4, 1)
    y_5 = torch.unsqueeze(y_5, 1)
    c = torch.cat((y_1, y_2, y_3, y_4, y_5), dim=1)
    d = torch.std(c, 1)
    consistency = torch.mean(d, 1)
    return consistency


def get_entropy(y_1, y_2, y_3, y_4, y_5):
    entropy1 = torch.sum(- y_1 * torch.log(y_1 + 1e-10), dim=1)
    entropy2 = torch.sum(- y_2 * torch.log(y_2 + 1e-10), dim=1)
    entropy3 = torch.sum(- y_3 * torch.log(y_3 + 1e-10), dim=1)
    entropy4 = torch.sum(- y_4 * torch.log(y_4 + 1e-10), dim=1)
    entropy5 = torch.sum(- y_5 * torch.log(y_5 + 1e-10), dim=1)
    entropy_norm = np.log(y_1.size(1))

    entropy = (entropy1 + entropy2 + entropy3 + entropy4 + entropy5) / (5 * entropy_norm)
    return entropy


def single_entropy(y_1):
    entropy1 = torch.sum(- y_1 * torch.log(y_1 + 1e-10), dim=1)
    entropy_norm = np.log(y_1.size(1))
    entropy = entropy1 / entropy_norm
    return entropy


def get_confidence(y_1, y_2, y_3, y_4, y_5):
    conf_1, indice_1 = torch.max(y_1, 1)
    conf_2, indice_2 = torch.max(y_2, 1)
    conf_3, indice_3 = torch.max(y_3, 1)
    conf_4, indice_4 = torch.max(y_4, 1)
    conf_5, indice_5 = torch.max(y_5, 1)
    confidence = (conf_1 + conf_2 + conf_3 + conf_4 + conf_5) / 5
    return confidence


# def norm(t):
#     mean = torch.mean(t)
#     var = torch.std(t) ** 2
#     t = (t - mean) / var
#     return t

def norm(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    return x
