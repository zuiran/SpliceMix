import os, sys, random, math, time, copy, shutil
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from utilities.coco import COCO2014
from utilities.voc import VOC2007, VOC2012
from utilities import utils_ddp
opj = os.path.join

def init(args):

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_devices).replace('[', '').replace(']', '')
    os.environ['MASTER_ADDR'] = args.master_addr
    if args.master_port == '17837':  # randomly selecting a prot
        import socket
        s=socket.socket()
        s.bind(("", 0))
        args.master_port = str(s.getsockname()[1])
        s.close()
    os.environ['MASTER_PORT'] = args.master_port
    args.world_size = len(str(args.cuda_devices).split(','))
    args.num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0])
    args.is_train = False if args.evaluate == 0 else True
    args.batch_size_per = args.batch_size // args.world_size

    set_seed(args.seed)

    args.model = args.model.replace('-', '_')
    model_name = args.model + (('_' + args.remark) if args.remark != '' else '')
    data_set = args.data_set

    args.save_path = opj(args.save_dir, data_set, model_name)
    args.start_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
    if utils_ddp.is_main_process():
        os.makedirs(opj(args.save_path, 'log'), exist_ok=True)
        file_name = model_name + f'_lr{args.lr:.1e}_' + args.start_time + '.py'
        shutil.copyfile(opj('models', args.model + '.py'),
                        opj(args.save_path, 'log', file_name))

    return args

def set_seed(seed=95):
    if seed > -1:
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            cudnn.deterministic = False  # If true, the speed may be low referring to https://pytorch.org/docs/stable/notes/randomness.html.
            torch.backends.cudnn.benchmark = True

def get_dataloader(train_set=None, test_set=None, args=None):
    # train_set, test_set, num_classes = get_dataset(args)
    pin_memory = False
    train_loader, test_loader = None, None
    if train_set != None:
        if sys.platform == 'linux':
            train_sampler = DistributedSampler(dataset=train_set, shuffle=True)
        else:
            train_sampler = None
        train_loader = DataLoader(train_set, batch_size=args.batch_size_per,
                                  num_workers=args.num_workers, pin_memory=pin_memory,
                                  drop_last=True, sampler=train_sampler, collate_fn=None)
    if test_set != None:
        if sys.platform == 'linux':
            test_sampler = DistributedSampler(dataset=test_set, shuffle=False)
        else:
            test_sampler = None
        test_loader = DataLoader(test_set, batch_size=args.batch_size_per,
                                num_workers=args.num_workers, pin_memory=pin_memory,
                                drop_last=False, sampler=test_sampler)
    return train_loader, test_loader

def get_dataset(args):
    data_dict = {'MS-COCO': COCO2014, 'VOC2007': VOC2007}
    test_transfm = get_transform(args, is_train=False)
    train_transfm = get_transform(args, is_train=True)

    if args.data_set in ('MS-COCO'):
        data_dir = opj(args.data_root, 'COCO2014')
        test_set = data_dict[args.data_set](data_dir, phase='val', transform=test_transfm)
        train_set = data_dict[args.data_set](data_dir, phase='train',
                                             transform=train_transfm)
    elif args.data_set in ('VOC2007'):
        data_dir = opj(args.data_root, 'VOC2007')
        test_set = data_dict[args.data_set](data_dir, phase='test',
                                            transform=test_transfm)
        train_set = data_dict[args.data_set](data_dir, phase='trainval',
                                            transform=train_transfm)

    num_classes = train_set.get_number_classes()
    return train_set, test_set, num_classes

def get_transform(args, is_train=True):

    if is_train:
        scale_size = args.image_size // 7
        transform = transforms.Compose([
            transforms.Resize((args.image_size + scale_size, args.image_size + scale_size)),
            MultiScaleCrop(args.image_size, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # Cutout(n_holes=1, scales=(0.3, 0.25, 0.1)),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return transform

def get_optimizer(args, model):

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.get_config_optim(args.lr, args.lrp),
                                    lr=args.lr,
                                    # betas=(0.99, 0.9),
                                    betas=(0.9, 0.999),
                                    # weight_decay=0)
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.get_config_optim(args.lr, args.lrp),
                                           lr=args.lr,
                                           betas=(0.9, 0.999),
                                           weight_decay=args.weight_decay)
    else:
        raise NotImplementedError('Only SGD or Adam can be chosen!')
    return optimizer

def get_lr_scheduler(args, optimizer):

    epoch_step = [e-1 for e in args.epoch_step]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=epoch_step, gamma=args.lr_decay)
    return scheduler

def get_learning_rate(optimizer):

    lr = [param_group['lr'] for param_group in optimizer.param_groups]
    return np.unique(lr)

def strftime(x):
    assert x >= 0

    if x < 3600:
        s = time.strftime('%M:%S', time.gmtime(x))
    elif x < 86400:
        s = time.strftime('%H:%M:%S', time.gmtime(x))
    elif x < 2678400:
        s = time.strftime('[%d] %H:%M:%S', time.gmtime(x))
    else:
        s = '!!So long time, can not be converted.!!'
    return s

def ignore_warning():  # not working
    import warnings
    warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step()` before `optimizer.step()`. ")

class MultiScaleCrop(object):
# from https://github.com/Yejin0111/ADD-GCN/blob/master/data/__init__.py
    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, 875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img):
        im_size = img.size
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
        ret_img_group = crop_img_group.resize((self.input_size[0], self.input_size[1]), self.interpolation)
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret

    def __str__(self):
        return self.__class__.__name__

if __name__ == '__main__':
    import os
    os.chdir('..')
    import main as m

    args = m.parser.parse_args()
    args.data_root = r'E:\PhD\Data_set\ImageSet'
    init(args)
    train_set, _, _ = get_dataset(args)
    train_loader, _ = get_dataloader(train_set, args=args)
    i, a = next(enumerate(train_loader))

    a = 'pause'


