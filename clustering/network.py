import math
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.functional import normalize

from resnet import ResNet, BasicBlock, Bottleneck


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.size(0),) + self.shape)
        
class Network(nn.Module):
    def __init__(self, encoder_size, inner_dim, class_num):
        super(Network, self).__init__()
        self.encoder_size = encoder_size
        self.inner_dim = inner_dim
        self.cluster_num = class_num
        self.kernel_size = int(math.sqrt(inner_dim / 16))
        self.encoder = Encoder(encoder_size)
        self.decoder = nn.Sequential(
            nn.Linear(10, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, inner_dim, bias=True),
            Reshape(16, self.kernel_size, self.kernel_size),
            nn.ConvTranspose2d(
                16, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
        self.instance_projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Softmax(dim=1)
        )
        self.cluster_layer = nn.Linear(10, class_num, bias=False) # class_num = 10
        self.reconstract = nn.Linear(512, 10)
        self.cluster_center = torch.rand([class_num, 10], requires_grad=False).cuda()
        
    def encode(self, x):
        x = self.encoder(x) # 256 * 512
        # x = self.reconstract(x) # 256 * 10
        # x = normalize(x)
        return x
    
    def decode(self, x):
        x = self.decoder(x)
        return x
    
    def forward_cluster(self, x):
        h = self.encode(x)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c
    
    def forward_clu(self, x):
        h = self.encode(x)
        n = self.reconstract(h)
        return n

    # 聚类层
    def cluster(self, z):
        return self.cluster_layer(z)

    def init_cluster_layer(self, alpha, cluster_center):
        self.cluster_layer.weight.data = 2 * alpha * cluster_center

    def compute_cluster_center(self, alpha):
        self.cluster_center = 1.0 / (2 * alpha) * self.cluster_layer.weight
        return self.cluster_center

    def normalize_cluster_center(self, alpha):
        self.cluster_layer.weight.data = (
            F.normalize(self.cluster_layer.weight.data, dim=1) * 2.0 * alpha
        )

    def predict(self, z):
        distance = torch.cdist(z, self.cluster_center, p=2)
        prediction = torch.argmin(distance, dim=1)
        return prediction

    def set_cluster_centroid(self, mu, cluster_id, alpha):
        self.cluster_layer.weight.data[cluster_id] = 2 * alpha * mu

class Encoder(nn.Module):
    def __init__(self, encoder_size=32, model_type='resnet18'):
        super(Encoder, self).__init__()

        # encoding block for local features
        print('Using a {}x{} encoder'.format(encoder_size, encoder_size))
        inplanes = 64
        if encoder_size == 32:
            conv1 = nn.Sequential(
                nn.Conv2d(3, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(inplanes),
                nn.ReLU(inplace=True)
            )
        elif encoder_size == 96 or encoder_size == 64:
            conv1 = nn.Sequential(
                nn.Conv2d(3, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(inplanes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            )

        else:
            raise RuntimeError("Could not build encoder."
                               "Encoder size {} is not supported".format(encoder_size))

        if model_type == 'resnet18':
            # ResNet18 block
            self.model = ResNet(BasicBlock, [2, 2, 2, 2], conv1)
        elif model_type == 'resnet34':
            self.model = ResNet(BasicBlock, [3, 4, 6, 3], conv1)
        elif model_type == 'resnet50':
            self.model = ResNet(Bottleneck, [3, 4, 6, 3], conv1)
        else:
            raise RuntimeError("Wrong model type")

        print(self.get_param_n())

    def get_param_n(self):
        w = 0
        for p in self.model.parameters():
            w += np.product(p.shape)
        return w

    def forward(self, x):
        return torch.flatten(self.model(x), 1)