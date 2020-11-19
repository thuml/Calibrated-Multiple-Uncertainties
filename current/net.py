from easydl import *
from torchvision import models


class BaseFeatureExtractor(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()

    def output_num(self):
        pass

    # def train(self, mode=True):
    #     # freeze BN mean and std
    #     for module in self.children():
    #         if isinstance(module, nn.BatchNorm2d):
    #             module.train(False)
    #         else:
    #             module.train(mode)


class ResNet50Fc(BaseFeatureExtractor):
    """
    ** input image should be in range of [0, 1]**
    """

    def __init__(self, model_path=None, normalize=True):
        super(ResNet50Fc, self).__init__()
        if model_path:
            if os.path.exists(model_path):
                self.model_resnet = models.resnet50(pretrained=False)
                self.model_resnet.load_state_dict(torch.load(model_path))
            else:
                raise Exception('invalid model path!')
        else:
            self.model_resnet = models.resnet50(pretrained=True)

        model_resnet = self.model_resnet
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.__in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features


class VGG16Fc(BaseFeatureExtractor):
    def __init__(self, model_path=None, normalize=True):
        super(VGG16Fc, self).__init__()
        if model_path:
            if os.path.exists(model_path):
                self.model_vgg = models.vgg16(pretrained=False)
                self.model_vgg.load_state_dict(torch.load(model_path))
            else:
                raise Exception('invalid model path!')
        else:
            self.model_vgg = models.vgg16(pretrained=True)

        if model_path or normalize:
            # pretrain model is used, use ImageNet normalization
            self.normalize = True
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            self.normalize = False

        model_vgg = self.model_vgg
        self.features = model_vgg.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i), model_vgg.classifier[i])
        self.feature_layers = nn.Sequential(self.features, self.classifier)

        self.__in_features = 4096

    def forward(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
        x = self.features(x)
        x = x.view(x.size(0), 25088)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self.__in_features


class CLS(nn.Module):
    """
    a two-layer MLP for classification
    """

    def __init__(self, in_dim, out_dim, bottle_neck_dim=256):
        super(CLS, self).__init__()
        self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
        # self.bottleneck_bn = nn.BatchNorm1d(bottle_neck_dim)
        self.fc = nn.Linear(bottle_neck_dim, out_dim)
        self.fc1 = nn.Linear(bottle_neck_dim, out_dim)
        self.fc2 = nn.Linear(bottle_neck_dim, out_dim)
        self.fc3 = nn.Linear(bottle_neck_dim, out_dim)
        self.fc4 = nn.Linear(bottle_neck_dim, out_dim)
        self.fc5 = nn.Linear(bottle_neck_dim, out_dim)
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.kaiming_uniform_(self.fc4.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc5.weight)

    def forward(self, x):
        feature = self.bottleneck(x)
        # feature = self.bottleneck_bn(feature)
        fc2_s = self.fc(feature)
        fc2_s2 = self.fc2(feature)
        fc2_s3 = self.fc3(feature)
        fc2_s4 = self.fc4(feature)
        fc2_s5 = self.fc5(feature)
        predict_prob = nn.Softmax(dim=-1)(fc2_s)

        return x, feature, fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5, predict_prob


class AdversarialNetwork(nn.Module):
    """
    AdversarialNetwork with a gredient reverse layer.
    its ``forward`` function calls gredient reverse layer first, then applies ``self.main`` module.
    """

    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

    def forward(self, x):
        x_ = self.grl(x)
        y = self.main(x_)
        return y
