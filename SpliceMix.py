import math
import torch
import numpy as np
import random
import torch.nn.functional as F
import torchvision.utils


class SpliceMix(object):
    def __init__(self, mode='SpliceMix', grids=('2x2',), n_grids=(0,), mix_prob=1.):
        # mode: 'SpliceMix' for custom grid setting; 'SpliceMix--Default=True' for default setting; 'SpliceMix--Mini=True' for minimalism setting
        # grids: grid settings, e.g., ['1x2', '2x2', '2x3-3']
        # n_grids: number of mixed samples in each setting, e.g., [3, 2, 1]
        # mix_prob: probability of using SpliceMix per mini-batch

        super(SpliceMix, self).__init__()

        self.Default = False
        self.Mini = False
        self.checkMode(mode)

        self.mix_prob = mix_prob
        self.grids = grids
        self.n_grids = n_grids
        self.use_asym = True
        self.config_default = {'1x2': .7, '2x2': .3, '2x3': .0, 'drop_rate': .3}

        self.mixer = self.Smix if self.Mini == False else self.Smix_minimalism
        if self.Default: print("SpliceMix w/ default setting")
        if self.Mini: print("SpliceMix w/ minimalism setting")

    def Smix(self, inputs, targets, ):
        if np.random.rand(1) > self.mix_prob:
            return inputs, targets, {}
        if self.Default:
            coin = random.random()
            coin_dp = random.random()
            self.n_grids = [inputs.shape[0]//4, ]
            ## defaut_max_2  84.78 SplicMix-CL, 84.11 SpliceMix in coco
            if coin > self.config_default['1x2']:
                n_drop = 1 if coin_dp < self.config_default['drop_rate'] else 0
                self.grids = [f'1x2-{n_drop}',]
            elif coin > self.config_default['2x2']:
                n_drop = random.sample(range(1, 4), 1)[0] if coin_dp < self.config_default['drop_rate'] else 0
                self.grids = [f'2x2-{n_drop}', ]
            elif coin > self.config_default['2x3']:
                n_drop = random.sample(range(1, 6), 1)[0] if coin_dp < self.config_default['drop_rate'] else 0
                self.grids = [f'2x3-{n_drop}', ]

        bs = inputs.shape[0]
        mix_ind = torch.zeros((bs), device=inputs.device)
        rand_ind = np.asarray([random.sample(range(bs), bs) for i in range(10)]).reshape(-1)  # enough for griding?
        mix_dict = {'rand_inds': [], 'rows': [], 'cols': [], 'n_drops': [], 'drop_inds': []}
        for g, ng in zip(self.grids, self.n_grids):
            g_row, g_col = [int(t) if '-' not in t else t.split('-') for t in g.split('x')]
            (g_col, n_drop) = [int(t) for t in g_col] if type(g_col) is list else (g_col, 0)
            g = g_row * g_col
            if ng == 0:
                if len(self.grids) == 1:
                    ng = bs // g
                else:
                    raise AssertionError('argument error, cannot execute c-mix')
            if ng * g > len(rand_ind):
                raise AssertionError('Too many mixed images to load, please enlarge #rand_ind')
            rand_ind_g = rand_ind[:ng * g]
            rand_ind = rand_ind[ng * g:]
            if g_row != g_col and self.use_asym:  # for asymmetric grids
                if np.random.randn() < 0: g_row, g_col = g_col, g_row
            inputs_mix_g, targets_mix_g, drop_ind = self.mix_fn(inputs[rand_ind_g], targets[rand_ind_g],
                                                       g_row=g_row, g_col=g_col, n_grid=ng, n_drop=n_drop,)

            inputs = torch.cat([inputs, inputs_mix_g], dim=0)
            targets = torch.cat([targets, targets_mix_g], dim=0)
            mix_dict['rand_inds'].append(rand_ind_g)
            mix_dict['rows'].append(g_row)
            mix_dict['cols'].append(g_col)
            mix_dict['n_drops'].append(n_drop)
            mix_dict['drop_inds'].append(drop_ind)  # the index in a mixed image, e.g., for a 2x2 grid, len(drop_ind)=4 || rand_ind_g[bool(drop_ind)] back to the index of dropped regular images
            mix_ind = torch.cat([mix_ind, torch.ones((ng), device=mix_ind.device)], dim=0)
        flag = {'mix_ind': mix_ind, 'mix_dict': mix_dict, }
        return inputs, targets, flag

    def mix_fn(self, inputs, targets, g_row, g_col, n_grid, n_drop=0):
        bs, c, h, w = inputs.shape
        g = g_row * g_col
        drop_ind = torch.zeros((bs), device=inputs.device)
        if n_drop > 0:
            drop_rand_ind = np.asarray([random.sample(range(i*g, (i+1)*g), n_drop) for i in range(n_grid)]).reshape(-1)
            drop_ind[drop_rand_ind] = 1
            inputs = inputs * (1 - drop_ind[:, None, None, None])
        inputs = F.interpolate(inputs, (h // g_row, w // g_col), mode='bilinear', align_corners=True)  # g*ng, C, h', w'
        inputs_mix = torchvision.utils.make_grid(inputs, nrow=g_col, padding=0)  # C, ng*h, w
        inputs_mix = inputs_mix.split(h//g_row * g_row, dim=1)  # tuple: ng, (C, h, w)
        inputs_mix = torch.stack(inputs_mix, dim=0)
        
        if (inputs_mix.shape[-2], inputs_mix.shape[-1]) != (h, w):
            inputs_mix = F.interpolate(inputs_mix, (h, w), mode='bilinear', align_corners=True)

        if n_drop > 0:
            targets = targets * (1 - drop_ind[:, None])
        targets_mix = targets.view(n_grid, g, -1).sum(1)  # ng, nc
        targets_mix[targets_mix > 0] = 1

        return inputs_mix, targets_mix, drop_ind

    def Smix_minimalism(self, X, Y):
        g_row, g_col = 2, 2
        B, C, H, W = X.shape
        ng = B // (g_row * g_col) * (g_row * g_col)
        Omega = random.sample(range(B), B//ng)
        X_ds = F.interpolate(X[Omega], (H // g_row, W // g_col), mode='bilinear', align_corners=True)  # g*ng, C, h', w'
        X_ = torchvision.utils.make_grid(X_ds, nrow=g_col, padding=0)  # C, ng*h, w
        X_ = X_.split(H, dim=1)  # tuple: ng, (C, h, w)
        X_ = torch.stack(X_, dim=0)  # ng, C, H, W
        Y_ = Y[Omega].view(ng, g_row * g_col, -1).sum(1)
        Y_[Y_ > 0] = 1

        X_hat = torch.cat((X, X_), dim=0)
        Y_hat = torch.cat((Y, Y_), dim=0)
        return X_hat, Y_hat, {}

    def checkMode(self, mode):
        if '--' in mode:  # like Splice--Mini=True
            str_list = mode.split('--')
            # mode = str_list[0]
            for s in str_list[1:]:
                exec(f"self.{s}")
        # return mode


def get_imgs(dir, bs=16, num_classes=10):
    transf = transforms.Compose([
        transforms.RandomResizedCrop(448, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    imgs = os.listdir(img_dir)
    inputs = torch.tensor([])

    for i in range(bs):
        img_path = os.path.join(dir, imgs[i])
        input = Image.open(img_path).convert('RGB')
        inputs = torch.cat((inputs, transf(input).unsqueeze(0)), 0)

    tgts = torch.rand(bs, num_classes)
    tgts[tgts>.7] = 1
    tgts[tgts<1] = 0
    return inputs, tgts

def plt_imgs(imgs, rows=2, tgts=None):
    plt.figure()
    for i in range(imgs.shape[0]):
        plt.subplot(rows, int(np.ceil(imgs.shape[0] / rows)), i + 1)
        plt.imshow(imgs[i].permute(1, 2, 0))
        plt.title(str(i))
        # plt.axis('off')
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_ticks([])
        frame1.axes.get_yaxis().set_ticks([])
        if tgts is not None:
            plt.xlabel(tgts[i])
    plt.show(block=False)

if __name__ == '__main__':
    import os
    import PIL.Image as Image
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms
    import numpy as np
    img_dir = 'E:\PhD\Data_set\ImageSet\VOC2007\VOCdevkit\VOC2007\JPEGImages'

    bs = 8
    imgs, ptgts = get_imgs(dir=img_dir, bs=bs)
    # imgs, ptgts = imgs.cuda(), ptgts.cuda()
    print(ptgts, ptgts.sum(-1))
    mixer = SpliceMix(mode='SpliceMix', grids=['1x2', '2x3-2'], n_grids=[1, 2]).mixer
    imgs_mix, tgts_mix, flag = mixer(imgs, ptgts)
    print(flag)
    print(tgts_mix[-5:])
    plt_imgs(imgs_mix.cpu(), tgts=tgts_mix.cpu().numpy())









