import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import Iterable
import math
import numpy as np

EPSILON = 1e-20


class TorchReshapeLayer(nn.Module):
    """
    make reshape operation a module that can be used in ``nn.Sequential``
    """
    def __init__(self, shape_without_batchsize):
        super(TorchReshapeLayer, self).__init__()
        self.shape_without_batchsize = shape_without_batchsize

    def forward(self, x):
       x = x.view(x.size(0), *self.shape_without_batchsize)
       return x


class TorchIdentityLayer(nn.Module):
    def __init__(self):
        super(TorchIdentityLayer, self).__init__()

    def forward(self, x):
       return x


class TorchLeakySoftmax(nn.Module):
    """
    leaky softmax, x_i = e^(x_i) / (sum_{k=1}^{n} e^(x_k) + coeff) where coeff >= 0

    usage::

        a = torch.zeros(3, 9)
        TorchLeakySoftmax().forward(a) # the output probability should be 0.1 over 9 classes

    """
    def __init__(self, coeff=1.0, dim=-1):
        super(TorchLeakySoftmax, self).__init__()
        self.softmax = torch.nn.Softmax(dim=dim)
        self.coeff = coeff
        self.dim = dim
        
    def forward(self, x):
        shape = list(x.size())
        shape[self.dim] = 1
        leaky = (torch.ones(*shape, dtype=x.dtype) * np.log(self.coeff)).to(x.device)
        concat = torch.cat([x, leaky], dim=self.dim)
        y = self.softmax(concat)
        prob_slicing = [slice(None, None, 1) for i in shape]
        prob_slicing[self.dim] = slice(None, -1, 1)
        prob = y[tuple(prob_slicing)]
        prob_slicing[self.dim] = slice(-1, None, 1)
        total_prob = 1.0 - y[tuple(prob_slicing)]
        return prob, total_prob


class TorchRandomProject(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(TorchRandomProject, self).__init__()
        self.register_buffer('matrix', Variable(torch.randn(1, out_dim, input_dim)))

    def forward(self, x):
        x = x.resize(x.size(0), 1, x.size(1))
        x = torch.sum(self.matrix * x, dim=-1)
        return x    


class GradientReverseLayer(torch.autograd.Function):
    """
    usage:(can't be used in nn.Sequential, not a subclass of nn.Module)::

        x = Variable(torch.ones(1, 2), requires_grad=True)
        grl = GradientReverseLayer.apply
        y = grl(0.5, x)

        y.backward(torch.ones_like(y))

        print(x.grad)

    """
    @staticmethod
    def forward(ctx, coeff, input):
        ctx.coeff = coeff
        # this is necessary. if we just return ``input``, ``backward`` will not be called sometimes
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        coeff = ctx.coeff
        return None, -coeff * grad_outputs


class GradientReverseModule(nn.Module):
    """
    wrap GradientReverseLayer to be a nn.Module so that it can be used in ``nn.Sequential``

    usage::

        grl = GradientReverseModule(lambda step : aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

        x = Variable(torch.ones(1), requires_grad=True)
        ans = []
        for _ in range(10000):
            x.grad = None
            y = grl(x)
            y.backward()
            ans.append(variable_to_numpy(x.grad))

        plt.plot(list(range(10000)), ans)
        plt.show() # you can see gradient change from 0 to -1
    """
    def __init__(self, scheduler):
        super(GradientReverseModule, self).__init__()
        self.scheduler = scheduler
        self.register_buffer('global_step', torch.zeros(1))
        self.coeff = 0.0
        self.grl = GradientReverseLayer.apply

    def forward(self, x):
        self.coeff = self.scheduler(self.global_step.item())
        if self.training:
            self.global_step += 1.0
        return self.grl(self.coeff, x)


class OptimizerManager:
    """
    automatic call op.zero_grad() when enter, call op.step() when exit
    usage::

        with OptimizerManager(op): # or with OptimizerManager([op1, op2])
            b = net.forward(a)
            b.backward(torch.ones_like(b))

    """
    def __init__(self, optims):
        self.optims = optims if isinstance(optims, Iterable) else [optims]

    def __enter__(self):
        for op in self.optims:
            op.zero_grad()

    def __exit__(self, exceptionType, exception, exceptionTraceback):
        for op in self.optims:
            op.step()
        self.optims = None # release reference, to avoid imexplicit reference
        if exceptionTraceback:
            print(exceptionTraceback)
            return False
        return True
    
    
class OptimWithSheduler:
    """
    usage::

        op = optim.SGD(lr=1e-3, params=net.parameters()) # create an optimizer
        scheduler = lambda step, initial_lr : inverseDecaySheduler(step, initial_lr, gamma=100, power=1, max_iter=100) # create a function
        that receives two keyword arguments:step, initial_lr
        opw = OptimWithSheduler(op, scheduler) # create a wrapped optimizer
        with OptimizerManager(opw): # use it as an ordinary optimizer
            loss.backward()
    """
    def __init__(self, optimizer, scheduler_func):
        self.optimizer = optimizer
        self.scheduler_func = scheduler_func
        self.global_step = 0.0
        for g in self.optimizer.param_groups:
            g['initial_lr'] = g['lr']

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        for g in self.optimizer.param_groups:
            g['lr'] = self.scheduler_func(step=self.global_step, initial_lr=g['initial_lr'])
        self.optimizer.step()
        self.global_step += 1


class TrainingModeManager:
    """
    automatic set and reset net.train(mode)
    usage::

        with TrainingModeManager(net, train=True): # or with TrainingModeManager([net1, net2], train=True)
            do whatever
    """
    def __init__(self, nets, train=False):
        self.nets = nets if isinstance(nets, Iterable) else [nets]
        self.modes = [net.training for net in nets]
        self.train = train

    def __enter__(self):
        for net in self.nets:
            net.train(self.train)

    def __exit__(self, exceptionType, exception, exceptionTraceback):
        for (mode, net) in zip(self.modes, self.nets):
            net.train(mode)
        self.nets = None # release reference, to avoid imexplicit reference
        if exceptionTraceback:
            print(exceptionTraceback)
            return False
        return True


def variable_to_numpy(x):
    """
    convert a variable to numpy, avoid too many parenthesis
    if the variable only contain one element, then convert it to python float(usually this is the test/train/dev accuracy)
    :param x:
    :return:
    """
    ans = x.cpu().data.numpy()
    if torch.numel(x) == 1:
        # make sure ans has no shape. (float requires number rather than ndarray)
        return float(np.sum(ans))
    return ans


def EntropyLoss(predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=EPSILON):
    """
    entropy for multi classification

    predict_prob should be size of [N, C]

    class_level_weight should be [1, C] or [N, C] or [C]

    instance_level_weight should be [N, 1] or [N]

    :param predict_prob:
    :param class_level_weight:
    :param instance_level_weight:
    :param epsilon:
    :return:
    """
    N, C = predict_prob.size()
    
    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'
        
    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    entropy = -predict_prob*torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * entropy * class_level_weight) / float(N)
