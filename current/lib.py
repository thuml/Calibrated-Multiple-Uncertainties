from easydl import *
import random
from torchvision.transforms.functional import _get_perspective_coeffs


def perspective(img, startpoints, endpoints, interpolation=Image.BICUBIC):
    """Perform perspective transform of the given PIL Image.

    Args:
        img (PIL Image): Image to be transformed.
        startpoints: List containing [top-left, top-right, bottom-right, bottom-left] of the orignal image
        endpoints: List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image
        interpolation: Default- Image.BICUBIC
    Returns:
        PIL Image:  Perspectively transformed Image.
    """

    coeffs = _get_perspective_coeffs(startpoints, endpoints)
    return img.transform(img.size, Image.PERSPECTIVE, coeffs, interpolation, fillcolor=(255, 255, 255))


class MyRandomPerspective(object):
    """Performs Perspective transformation of the given PIL Image randomly with a given probability.

    Args:
        interpolation : Default- Image.BICUBIC

        p (float): probability of the image being perspectively transformed. Default value is 0.5

        distortion_scale(float): it controls the degree of distortion and ranges from 0 to 1. Default value is 0.5.

    """

    def __init__(self, distortion_scale=0.5, p=0.5, interpolation=Image.BICUBIC):
        self.p = p
        self.interpolation = interpolation
        self.distortion_scale = distortion_scale

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be Perspectively transformed.

        Returns:
            PIL Image: Random perspectivley transformed image.
        """

        if random.random() < self.p:
            width, height = img.size
            startpoints, endpoints = self.get_params(width, height, self.distortion_scale)
            return perspective(img, startpoints, endpoints, self.interpolation)
        return img

    @staticmethod
    def get_params(width, height, distortion_scale):
        """Get parameters for ``perspective`` for a random perspective transform.

        Args:
            width : width of the image.
            height : height of the image.

        Returns:
            List containing [top-left, top-right, bottom-right, bottom-left] of the original image,
            List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image.
        """
        half_height = int(height / 2)
        half_width = int(width / 2)
        topleft = (random.randint(0, int(distortion_scale * half_width)),
                   random.randint(0, int(distortion_scale * half_height)))
        topright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(0, int(distortion_scale * half_height)))
        botright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        botleft = (random.randint(0, int(distortion_scale * half_width)),
                   random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        startpoints = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


def reverse_sigmoid(y):
    return torch.log(y / (1.0 - y + 1e-10) + 1e-10)


def get_label_weight(label, common_classes):
    weight = [0.0 for i in range(len(label))]
    for i in range(len(label)):
        if label[i] in common_classes:
            weight[i] = 1.0
    return torch.tensor(weight)


def get_source_share_weight(domain_out, before_softmax, domain_temperature=1.0, class_temperature=10.0):
    before_softmax = before_softmax / class_temperature
    after_softmax = nn.Softmax(-1)(before_softmax)
    domain_logit = reverse_sigmoid(domain_out)
    domain_logit = domain_logit / domain_temperature
    domain_out = nn.Sigmoid()(domain_logit)

    entropy = torch.sum(- after_softmax * torch.log(after_softmax + 1e-10), dim=1, keepdim=True)
    entropy_norm = entropy / np.log(after_softmax.size(1))
    weight = entropy_norm - domain_out
    weight = weight.detach()
    return weight


def get_entropy(fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5, domain_temperature=1.0, class_temperature=10.0):
    fc2_s = nn.Softmax(-1)(fc2_s)
    fc2_s2 = nn.Softmax(-1)(fc2_s2)
    fc2_s3 = nn.Softmax(-1)(fc2_s3)
    fc2_s4 = nn.Softmax(-1)(fc2_s4)
    fc2_s5 = nn.Softmax(-1)(fc2_s5)

    entropy = torch.sum(- fc2_s * torch.log(fc2_s + 1e-10), dim=1)
    entropy2 = torch.sum(- fc2_s2 * torch.log(fc2_s2 + 1e-10), dim=1)
    entropy3 = torch.sum(- fc2_s3 * torch.log(fc2_s3 + 1e-10), dim=1)
    entropy4 = torch.sum(- fc2_s4 * torch.log(fc2_s4 + 1e-10), dim=1)
    entropy5 = torch.sum(- fc2_s5 * torch.log(fc2_s5 + 1e-10), dim=1)
    entropy_norm = np.log(fc2_s.size(1))

    weight = (entropy + entropy2 + entropy3 + entropy4 + entropy5) / (5 * entropy_norm)
    return weight


def get_consistency(fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5):
    fc2_s = nn.Softmax(-1)(fc2_s)
    fc2_s2 = nn.Softmax(-1)(fc2_s2)
    fc2_s3 = nn.Softmax(-1)(fc2_s3)
    fc2_s4 = nn.Softmax(-1)(fc2_s4)
    fc2_s5 = nn.Softmax(-1)(fc2_s5)

    fc2_s = torch.unsqueeze(fc2_s, 1)
    fc2_s2 = torch.unsqueeze(fc2_s2, 1)
    fc2_s3 = torch.unsqueeze(fc2_s3, 1)
    fc2_s4 = torch.unsqueeze(fc2_s4, 1)
    fc2_s5 = torch.unsqueeze(fc2_s5, 1)
    c = torch.cat((fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5), dim=1)
    d = torch.std(c, 1)
    consistency = torch.mean(d, 1)
    return consistency


def get_predict_prob(fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5):
    fc2_s = nn.Softmax(-1)(fc2_s)
    fc2_s2 = nn.Softmax(-1)(fc2_s2)
    fc2_s3 = nn.Softmax(-1)(fc2_s3)
    fc2_s4 = nn.Softmax(-1)(fc2_s4)
    fc2_s5 = nn.Softmax(-1)(fc2_s5)

    fc2_s = torch.unsqueeze(fc2_s, 1)
    fc2_s2 = torch.unsqueeze(fc2_s2, 1)
    fc2_s3 = torch.unsqueeze(fc2_s3, 1)
    fc2_s4 = torch.unsqueeze(fc2_s4, 1)
    fc2_s5 = torch.unsqueeze(fc2_s5, 1)
    c = torch.cat((fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5), dim=1)
    predict_prob = torch.mean(c, 1)
    predict_prob = nn.Softmax(-1)(predict_prob)
    return predict_prob


def get_target_weight(entropy, consistency, threshold):
    sorce = (entropy + consistency) / 2
    weight = [0.0 for i in range(len(sorce))]
    for i in range(len(sorce)):
        if sorce[i] < (threshold / 2):
            weight[i] = 1.0
    return torch.tensor(weight)


def normalize_weight(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    return x.detach()


def nega_weight(x):
    # min_val = x.min()
    # max_val = x.max()
    # x = 1 - (x - min_val) / (max_val - min_val)
    x = 1 - x
    return x.detach()


def nega_normalize_weight(x):
    min_val = x.min()
    max_val = x.max()
    x = 1 - (x - min_val) / (max_val - min_val)
    x = 1 - x
    return x.detach()


def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
