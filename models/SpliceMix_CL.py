import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import models.loss_fns as loss_fns

class model(nn.Module):
    def __init__(self, num_classes, pretrained=True, args=None):
        super(model, self).__init__()
        M = torchvision.models.resnet101(pretrained=pretrained)
        # res = torchvision.models.resnet101(pretrained=pretrained, norm_layer=lib.FrozenBatchNorm2d)
        # res = torchvision.models.resnet50(True)
        self.backbone = nn.Sequential(M.conv1, M.bn1, M.relu, M.maxpool,
                                      M.layer1, M.layer2, M.layer3, M.layer4, )
        self.num_classes = num_classes

        self.glb_pooling = nn.AdaptiveMaxPool2d((1, 1))
        self.cls = nn.Linear(M.layer4[-1].conv3.out_channels, num_classes)

    def forward(self, inputs, args=None):  #

        feas = self.backbone(inputs)  # bs, C, 28, 28
        fea_gp = self.glb_pooling(feas).flatten(1)  # bs, C
        preds = self.cls(fea_gp)

        if self.training:  ## split feature maps
            mix_ind, mix_dict = args['flag']['mix_ind'], args['flag']['mix_dict']
            feas_r, preds_r = feas[(1 - mix_ind).bool()], preds[(1 - mix_ind).bool()]
            feas_m, _ = feas[mix_ind.bool()], preds[mix_ind.bool()]
            bs_m, C, h, w = feas_m.shape

            ng_list = []
            preds_m = preds_m_r = torch.tensor([], device=inputs.device)
            for i, (rand_ind, g_row, g_col, n_drop, drop_ind) in enumerate(zip(mix_dict['rand_inds'], mix_dict['rows'], mix_dict['cols'], mix_dict['n_drops'], mix_dict['drop_inds'])):  # insertion ordered in Dict after python 3.6 -> better code to be done
                ng = len(rand_ind) // (g_row * g_col)
                fea_m = feas_m[sum(ng_list): sum(ng_list) + ng]
                ng_list.append(ng)
                # fea_r, tgt_r = feas_r[rand_ind], None
                if h % g_row + w % g_col != 0:
                    fea_m = F.interpolate(fea_m, (h // g_row * g_row, w // g_col * g_col), mode='bilinear', align_corners=True)
                chunks = [c.split(fea_m.shape[-1] // g_col, dim=-1) for c in fea_m.split(fea_m.shape[-2] // g_row, dim=-2)]  # [[[]\in{ng, C, h//g_row(h'), w//g_col(w')},[],...](sub-imgs in row 1), [[],[],...](sub-imgs in row 2), ...]
                fea_m = torch.stack([torch.stack(c, dim=1) for c in chunks], dim=1)  # ng, g_row, g_col, C, h', w' || stack in cols per row, then stack in rows
                fea_m = fea_m.view(-1, C, fea_m.shape[-2], fea_m.shape[-1])  # ng, C, h, w

                # pred_m_r = preds_r[rand_ind] * (1 - drop_ind[:, None]) if n_drop > 0 else preds_r[rand_ind]
                pred_m_r = torch.masked_fill(preds_r[rand_ind], drop_ind[:, None]==1, -1e3)


                fea_m_gp = self.glb_pooling(fea_m).flatten(1)
                pred_m = self.cls(fea_m_gp)
                pred_m = torch.masked_fill(pred_m, drop_ind[:, None]==1, -1e3)

                preds_m = torch.cat((preds_m, pred_m), dim=0)
                preds_m_r = torch.cat((preds_m_r, pred_m_r), dim=0)

            preds = (preds, preds_m, preds_m_r)
            # return preds[:(1 - mix_ind).sum()], ...
        return preds


    def splitting(self, fea, mix_dict):
        pass

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
        self.bce = self.loss_fn

    def forward(self, inputs, targets):
        if len(inputs) == 3:
            preds, preds_m, preds_m_r = inputs
            loss_bce = self.bce(preds, targets)
            if targets.shape[-1] == 20:  # for VOC2007
                loss_cl = self.bce(preds_m, preds_m_r.sigmoid())
            else:  # for MS-COCO and others
                loss_cl = self.bce(preds_m, preds_m_r.sigmoid().detach())
            loss = loss_bce + loss_cl
        else:
            loss = self.bce(inputs, targets)
        return loss

if __name__ == '__main__':
    from SpliceMix import SpliceMix

    bs = 8
    inputs = torch.randn((bs, 3, 224, 224)).cuda()
    target = torch.zeros((bs, 20)).cuda()
    target[:, 1:3] = 1

    mixer = SpliceMix(mode='cmix', grids=['1x2', '2x3-2'], n_grids=[1, 2]).mixer
    imgs_mix, tgts_mix, flag = mixer(inputs, target)
    args = {'flag': flag}

    loss_fn = Loss_fn()

    model = model(20).cuda()
    output = model(imgs_mix, args)

    loss = loss_fn(output, tgts_mix)
    loss.backward()

    a= 'pause'
