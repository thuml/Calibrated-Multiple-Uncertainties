import pathlib2
import os
import numpy as np
import re


def getHomePath():
    return str(pathlib2.Path.home())


def join_path(*a):
    return os.path.join(*a)


def ZipOfPython3(*args):
    """
    return a iter rather than a list

    each of args should be iterable

    usage::

        def f(i):
            while True:
                yield i
                i += 1

        for x in ZipOfPython3(f(1), f(2), f(3)):
            print(x)

    python2 zip will stuck in this infinite generator function, so we should use ZipOfPython3

    another situation is the function returns a datapoint, and zip will make it a list,
    meaning the whole dataset will be loaded into the memory, which is not desirable
    """
    args = [iter(x) for x in args]
    while True:
        yield [next(x) for x in args]


class AccuracyCounter:
    """
    in supervised learning, we often want to count the test accuracy.
    but the dataset size maybe is not dividable by batch size, causing a remainder fraction which is annoying.
    also, sometimes we want to keep trace with accuracy in each mini-batch(like in train mode)
    this class is a simple class for counting accuracy.

    usage::

        counter = AccuracyCounter()
        iterate over test set:
            counter.addOneBatch(predict, label) -> return accuracy in this mini-batch
        counter.reportAccuracy() -> return accuracy over whole test set
    """
    def __init__(self):
        self.Ncorrect = 0.0
        self.Npred = 0.0
        self.Ntotal = 0.0

    def addOneBatch(self, predict, label):
        assert predict.shape == label.shape
        correct_prediction = np.equal(np.argmax(predict, 1), np.argmax(label, 1))
        Ncorrect = np.sum(correct_prediction.astype(np.float32))
        Ntotal = len(label)
        self.Ncorrect += Ncorrect
        self.Ntotal += Ntotal
        return Ncorrect / Ntotal
    
    def reportAccuracy(self):
        """
        :return: **return nan when 0 / 0**
        """
        return np.asarray(self.Ncorrect, dtype=float) / np.asarray(self.Ntotal, dtype=float)


class Accumulator(dict):
    """
    accumulate data and store them in a dict

    usage::

        with Accumulator(['weight', 'coeff']) as accumulator:
            for data in data_generator():
                # forward ......
                weight = xxx
                coeff = xxx

                accumulator.updateData(scope=globals())

        # do whatever with accumulator['weight'] and accumulator['coeff']

    """

    def __init__(self, name_or_names, accumulate_fn=np.concatenate):
        super(Accumulator, self).__init__()
        self.names = [name_or_names] if isinstance(name_or_names, str) else name_or_names
        self.accumulate_fn = accumulate_fn
        for name in self.names:
            self.__setitem__(name, [])

    def updateData(self, scope):
        for name in self.names:
            self.__getitem__(name).append(scope[name])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_tb:
            print(exc_tb)
            return False

        for name in self.names:
            self.__setitem__(name, self.accumulate_fn(self.__getitem__(name)))

        return True


def sphere_sample(size):
    '''
    sample noise from high dimensional gaussian distribution and project it to high dimension sphere of unit ball(with L2 norm = 1)
    '''
    z = np.random.normal(size=size)
    z = z / (np.sqrt(np.sum(z**2, axis=1, keepdims=True)) + 1e-6)
    return z.astype(np.float32)


def sphere_interpolate(a, b, n=64):
    '''
    interpolate along high dimensional ball, according to <Sampling Generative Networks> 
    '''
    a = a / (np.sqrt(np.sum(a**2)) + 1e-6)
    b = b / (np.sqrt(np.sum(b**2)) + 1e-6)
    dot = np.sum(a * b)
    theta = np.arccos(dot)
    mus = [x * 1.0 / (n - 1) for x in range(n)]
    ans = np.asarray([(np.sin((1.0 - mu) * theta) * a + np.sin(mu * theta) * b) / np.sin(theta) for mu in mus], dtype=np.float32)
    return ans


def mergeImage_color(images, rows, cols=None):
    '''
    images:(np.ndarray) 4D array, [N, H, W, C], C = 3
    '''
    cols = rows if not cols else cols
    size = (rows, cols)
    h, w, c = images.shape[1], images.shape[2], images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c), dtype=images.dtype) # use images.dtype to keep the same dtype
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img


def mergeImage_gray(images, rows, cols=None):
    '''
    images:(np.ndarray) 3D array, [N, H, W], C = 3
    '''
    cols = rows if not cols else cols
    size = (rows, cols)
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]), dtype=images.dtype) # use images.dtype to keep the same dtype
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image
    return img


def to_gray_np(img):
    '''
    normalize a img to lie in the range of [0, 1], turning it into a gray image
    '''
    img = img.astype(np.float32)
    img = ((img - img.min()) / (img.max() - img.min()) * 1).astype(np.float32)
    return img


def to_rgb_np(img):
    '''
    normalize a img to lie in the range of [0, 255], turning it into a RGB image
    '''
    img = img.astype(np.float32)
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    return img


def getID():
    '''
    return an unique id on each call (useful when you need a sequence of unique identifiers)
    '''
    getID.x += 1
    return getID.x
getID.x = 0


def is_in_notebook():
    import sys
    return 'ipykernel' in sys.modules


def clear_output():
    """
    clear output for both jupyter notebook and the console
    """
    import os
    os.system('cls' if os.name == 'nt' else 'clear')
    if is_in_notebook():
        from IPython.display import clear_output as clear
        clear()


class Nonsense(object):
    """
    placeholder class to support a.b.c().e.f.g.h with nonsense value
    """

    def __getattr__(self, item):
        if item not in self.__dict__:
            self.__dict__[item] = Nonsense()
        return self.__dict__[item]

    def __call__(self, *args, **kwargs):
        return Nonsense()

    def __str__(self):
        return "Nonsense object!"

    def __repr__(self):
        return "Nonsense object!"


def clear_and_mkdir(dir):
    os.system('rm %s -r' % dir)
    os.system('mkdir %s' % dir)