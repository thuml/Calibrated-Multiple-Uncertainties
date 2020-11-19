import numpy as np

def inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000):
    '''
    change as initial_lr * (1 + gamma * min(1.0, iter / max_iter) ) ** (- power)
    as known as inv learning rate sheduler in caffe,
    see https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto

    the default gamma and power come from <Domain-Adversarial Training of Neural Networks>

    code to see how it changes(decays to %20 at %10 * max_iter under default arg)::

        from matplotlib import pyplot as plt

        ys = [inverseDecaySheduler(x, 1e-3) for x in range(10000)]
        xs = [x for x in range(10000)]

        plt.plot(xs, ys)
        plt.show()

    '''
    return initial_lr * ((1 + gamma * min(1.0, step / float(max_iter))) ** (- power))


def aToBSheduler(step, A, B, gamma=10, max_iter=10000):
    '''
    change gradually from A to B, according to the formula (from <Importance Weighted Adversarial Nets for Partial Domain Adaptation>)
    A + (2.0 / (1 + exp(- gamma * step * 1.0 / max_iter)) - 1.0) * (B - A)

    =code to see how it changes(almost reaches B at %40 * max_iter under default arg)::

        from matplotlib import pyplot as plt

        ys = [aToBSheduler(x, 1, 3) for x in range(10000)]
        xs = [x for x in range(10000)]

        plt.plot(xs, ys)
        plt.show()

    '''
    ans = A + (2.0 / (1 + np.exp(- gamma * step * 1.0 / max_iter)) - 1.0) * (B - A)
    return float(ans)