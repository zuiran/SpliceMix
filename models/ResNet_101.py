import torch
import torchvision
import torch.nn as nn
import models.loss_fns as loss_fns

class model(nn.Module):
    def __init__(self, num_classes, pretrained=True, args=None):
        super(model, self).__init__()
        M = torchvision.models.resnet101(pretrained=pretrained)
        self.backbone = nn.Sequential(M.conv1, M.bn1, M.relu, M.maxpool,
                                      M.layer1, M.layer2, M.layer3, M.layer4, )
        self.num_classes = num_classes

        self.glb_pooling = nn.AdaptiveMaxPool2d((1, 1))
        self.cls = nn.Linear(M.layer4[-1].conv3.out_channels, num_classes)

    def forward(self, inputs, args=None):  #

        fea4 = self.backbone(inputs)  # bs, C, h, w
        fea_gmp = self.glb_pooling(fea4).flatten(1)  # bs, C
        output = self.cls(fea_gmp)    # bs, nc

        return output

    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.backbone.parameters()))
        large_lr_layers = filter(lambda p: id(p) not in small_lr_layers, self.parameters())
        return [
            {'params': self.backbone.parameters(), 'lr': lr * lrp},
            {'params': large_lr_layers, 'lr': lr},
        ]

class Loss_fn(loss_fns.BCELoss):
    def __init__(self):
        super(Loss_fn, self).__init__()

if __name__ == '__main__':
    inputs = torch.randn((2, 3, 448, 448)).cuda()
    target = torch.zeros((2, 20)).cuda()
    target[:, 1:3] = 1

    loss_fn = Loss_fn()

    model = model(20).cuda()
    output = model(inputs)

    loss = loss_fn(output, target)
    loss.backward()

    a= 'pause'
