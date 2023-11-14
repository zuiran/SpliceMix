import torch
import torch.nn as nn
import time, os, shutil
from torch.cuda.amp import GradScaler, autocast
from utilities import utils, metric, utils_ddp, warmup, logger
import SpliceMix
import models


class Engine(object):
    def __init__(self, args):
        super(Engine, self).__init__()
        self.args = args
        self.result = {}
        self.result['train'] = {'epoch': [], 'lr': [], 'precision': {'mAP': [], 'AP': []}, 'loss': []}
        self.result['val'] = {'epoch': [], 'lr': [], 'precision': {'mAP': [], 'AP': []}, 'loss': []}
        self.result['val_best'] = {'epoch': 0, 'precision': {'mAP': 0., 'AP': [0]}, 'loss': -1.}

        self.meter = {}
        self.reset_meters()

        self.rank = utils_ddp.get_rank()
        log_file = self.args.model + self.args.remark + f'_lr{args.lr:.1e}_' + self.args.start_time+ '.log'
        self.logger = logger.setup_logger(os.path.join(self.args.save_path, 'log', log_file), self.rank)
        self.logger.info(args)

        # utils.ignore_warning()  # not working
        self.init()

    def init(self):
        train_set, test_set, self.args.num_classes = utils.get_dataset(self.args)
        self.dataset = {'train': train_set, 'test': test_set}
        self.scaler = GradScaler(enabled=not self.args.disable_amp)

        args = {}
        self.model = getattr(models, self.args.model).model(self.args.num_classes, args=args).to(self.rank)
        self.optimizer = utils.get_optimizer(self.args, self.model)

        self.loss_fn = getattr(models, self.args.model).Loss_fn().to(self.rank)

        self.train_loader, self.test_loader = utils.get_dataloader(train_set=self.dataset['train'],
                                                       test_set=self.dataset['test'], args=self.args)
        if self.args.warmup_epochs > 0:
            self.warmup_scheduler = warmup.WarmUpLR(self.optimizer,
                                                    total_iters=len(self.train_loader) * self.args.warmup_epochs)
        self.lr_scheduler = utils.get_lr_scheduler(self.args, self.optimizer)
        self.load_checkpoint()


        self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.rank])  # , broadcast_buffers=False
        # torch.nn.parallel.DistributedDataParallel.find_unused_parameters=True
        # self.model.find_unused_parameters=True
        if 'SpliceMix' in self.args.mixer:
            self.mixer = SpliceMix.SpliceMix(mode=self.args.mixer, grids=self.args.grids,
                                             n_grids=self.args.n_grids, mix_prob=self.args.Sprob).mixer

    def train(self):

        if self.args.start_epoch == 0:
            self.args.start_epoch = 1
        for epoch in range(self.args.start_epoch, self.args.epochs+1):
            train_loader = self.train_loader
            self.model.train()
            self.on_start_epoch(epoch)
            train_loader.sampler.set_epoch(epoch)
            torch.cuda.empty_cache()

            for i, data in enumerate(train_loader):
                inputs, targets, targets_gt, file_name = self.on_start_batch(data)
                outputs, loss = self.on_forward(inputs, targets, file_name, is_train=True)
                self.on_end_batch(outputs, targets_gt.data, loss.data, file_name)

            self.on_end_epoch(is_train=True, result=self.result['train'])

            self.lr_scheduler.step()

            if self.args.evaluate > 0 and ((epoch % self.args.evaluate == 0) or epoch == 1):
                self.evaluate(epoch=epoch)

    def evaluate(self, epoch=0):
        torch.cuda.empty_cache()
        val_loader = self.test_loader

        self.model.eval()
        self.on_start_epoch(epoch)
        interval = 0
        for i, data in enumerate(val_loader):
            count = (epoch-1) * (len(val_loader)+interval) + i
            # self.count = count

            inputs, targets, targets_gt, file_name = self.on_start_batch(data)
            outputs, loss = self.on_forward(inputs, targets, file_name, is_train=False)

            self.on_end_batch(outputs, targets_gt.data, loss.data, file_name)

        self.on_end_epoch(is_train=False, result=self.result['val'], result_best=self.result['val_best'])

    def on_forward(self, inputs, targets, file_name, is_train,):

        args = {}
        if is_train:
            with autocast(enabled=not self.args.disable_amp):
                if 'SpliceMix' in self.args.mixer:
                    inputs, targets, flag = self.mixer(inputs, targets)

                if self.args.model in ['SpliceMix_CL']: args = {'flag': flag,}
                outputs = self.model(inputs, args)

                loss = self.loss_fn(outputs, targets)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            if self.args.disable_amp:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.args.warmup_epochs > 0 and self.epoch <= self.args.warmup_epochs:
                self.warmup_scheduler.step()
        else:
            model = self.model
            with torch.no_grad():
                with autocast(enabled=not self.args.disable_amp):
                    outputs = model(inputs, args)
                    loss = self.loss_fn(outputs, targets)

        outputs = outputs[0][:inputs.shape[0]].data if type(outputs) == tuple else outputs[:inputs.shape[0]].data
        return outputs, loss

    def on_start_batch(self, data):

        inputs = data['image'].to(self.rank)  # , non_blocking=True
        targets_gt = data['target']
        file_name = data['name']
        targets = targets_gt.clone().to(self.rank)  # , non_blocking=True
        targets[targets == -1] = 0

        return inputs, targets, targets_gt, file_name

    def on_end_batch(self, outputs, targets_gt, loss, image_name=''):

        bs = self.args.batch_size

        # if utils_ddp.is_main_process():
        outputs = utils_ddp.distributed_concat(outputs.detach(), bs)
        targets_gt = utils_ddp.distributed_concat(targets_gt.detach().to(self.rank), bs)
        loss_all = utils_ddp.distributed_concat(loss.detach().unsqueeze(0), utils_ddp.get_world_size())

        # utils_ddp.barrier()
        self.meter['loss'].add(loss.cpu())
        if utils_ddp.is_main_process():
            self.meter['loss_all'].add(loss_all.detach().cpu().mean())
            self.meter['ap'].add(outputs.detach().cpu(), targets_gt.cpu(), image_name)  # TODO: image_name is unused

    def on_start_epoch(self, epoch):
        self.epoch = epoch
        self.epoch_time = time.time()
        self.reset_meters()

    def on_end_epoch(self, is_train, result, result_best=None):

        self.lr_curr = utils.get_learning_rate(self.optimizer)

        self.epoch_time = time.time() - self.epoch_time
        meter = self.meter
        loss = meter['loss'].average()
        if utils_ddp.is_main_process():
            loss_all = meter['loss_all'].average()
            (mAP, AP) = meter['ap'].mAP() if not is_train else (-1., torch.zeros(1))
            OP, OR, OF1, CP, CR, CF1 = meter['ap'].overall()
        else:
            loss_all = torch.tensor(-1)
            mAP, AP = -1, torch.zeros(1) - 1
            OP, OR, OF1, CP, CR, CF1 = (-1 for i in range(6))
        # self.logger.info('end mAP')
        utils_ddp.barrier()

        result['precision']['mAP'].append(mAP)
        result['precision']['AP'].append(AP)
        result['epoch'].append(self.epoch)
        result['lr'].append(self.lr_curr)
        result['loss'].append(loss_all.item())
        str_precision = f'OF1:{OF1:.2f}, CF1:{CF1:.2f}' + (f', mAP:{mAP:.4f}' if mAP != -1 else '')

        is_best = False
        if is_train:
            str_end_epoch = f'[Epoch {self.epoch}, lr{self.lr_curr}] ' \
                            f'[Train] elapsed time:{utils.strftime(self.epoch_time)}s, loss: {loss:.4f}, {str_precision} .' \
                # + f' Acc: {acc:.4f}' + f'   | {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            self.logger.info(str_end_epoch)
        else:
            str_prefix = '--'  # + '[logged]' if 'log' in dir(self) and self.args.evaluate != 0 else ''
            str_val = str_prefix + '[Test]'
            str_end_epoch = str_val + f' elapsed time: {utils.strftime(self.epoch_time)}s, ' \
                                                   f'loss: {loss:.4f}, {str_precision} .'
            if self.args.print_verbose == 2 or self.args.evaluate == 0:
                OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = meter['ap'].overall_topk(self.args.acc_top_k)
                str_verbose = f'\nOP: {OP:.2f}, OR: {OR:.2f}, OF1: {OF1:.2f}, CP: {CP:.2f}, CR: {CR:.2f}, CF1: {CF1:.2f}, ' \
                              f'OP_{self.args.acc_top_k}: {OP_k:.2f}, OR_{self.args.acc_top_k}: {OR_k:.2f}, ' \
                              f'OF1_{self.args.acc_top_k}: {OF1_k:.2f}, CP_{self.args.acc_top_k}: {CP_k:.2f}, ' \
                              f'CR_{self.args.acc_top_k}: {CR_k:.2f}, CF1_{self.args.acc_top_k}: {CF1_k:.2f} \n {AP}.'
            elif self.args.print_verbose == 1:
                str_verbose = f'\nOP: {OP:.2f}, OR: {OR:.2f}, OF1: {OF1:.2f}, CP: {CP:.2f}, CR: {CR:.2f}, CF1: {CF1:.2f} .'
            else:
                str_verbose = ''
            str_end_epoch += str_verbose
            self.logger.info(str_end_epoch)

            if result_best['precision']['mAP'] < mAP:
                is_best = True
                result_best['precision'] = {'mAP': mAP, 'AP': AP}
                result_best['epoch'] = self.epoch
                result_best['loss'] = loss

            str_val_best = f"--[Test-best] (E{self.result['val_best']['epoch']}, " \
                           f"L{self.result['val_best']['loss']:.4f}), " \
                           f"mAP: {self.result['val_best']['precision']['mAP']:.4f}"
            str_val_best += ' .'
            self.logger.info(str_val_best)

        if self.args.evaluate != 0 and utils_ddp.is_main_process():
            self.save_checkpoint(is_train, is_best)
            # self.save_result(is_train=is_train, is_best=is_best)
        if self.args.evaluate == 0 and utils_ddp.is_main_process():
            self.save_result(is_train=is_train)
        utils_ddp.barrier()

        if not is_train: return result_best['precision']['mAP']

    def save_checkpoint(self, is_train, is_best):
        opj = os.path.join

        file = f'ChkpotLast_L{self.args.lr:.1e}_{self.args.model}.pt'
        result = self.result['val']
        result_best = self.result['val_best']
        state_dict = self.model.module.state_dict()

        if is_train:
            result = self.result['train']

        checkpoint = {'epoch': self.epoch,
                      'lr_curr': self.lr_curr,
                      'model_state_dict': state_dict,
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'result': self.result,
                      'mAP_curr': result['precision']['mAP'][-1],
                      'AP_curr': result['precision']['AP'][-1],
                      'loss_val_curr': result['loss'][-1],
                      'mAP_best': result_best['precision']['mAP'],
                      'AP_best': result_best['precision']['AP'],
                      'epoch_best': result_best['epoch'],
                      'loss_val_best': result_best['loss'],
                      'loss_tr': self.result['train']['loss'][-1],
                      'args': self.args,
                      }

        if is_best:
            file_best = f'ChkpotBest_L{self.args.lr:.1e}_{self.args.model}.pt'
            file_best_path = opj(self.args.save_path, file_best)
            torch.save(checkpoint, file_best_path)

    def load_checkpoint(self):

        if self.args.resume == '':
            return
        else:
            file = self.args.resume
            map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
            checkpoint = torch.load(file, map_location=map_location)
        try:
            if self.args.start_epoch == 0 :
                self.args.start_epoch = checkpoint['epoch'] + 1
            self.result = checkpoint['result']
        except:
            self.logger.info('checkpoint-Dict keys are not matched')

        try:
            checkpoint['model_state_dict'] = self.convertDict_state(checkpoint['model_state_dict'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except:
            state_dict = self.model.state_dict()
            # pretrained_dict = {}
            pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in state_dict and k != 'cls.weight'
                             and k != 'cls.bias'}
            state_dict.update(pretrained_dict)
            self.model.load_state_dict(state_dict)
            self.logger.info(f'can not fully load checkpoint, try to load partly. {len(pretrained_dict.keys())}/{len(state_dict.keys())}')
        if self.args.load_optimizer:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # self.optimizer.param_groups = self.model.get_config_optim(self.args.lr, self.args.lrp)
                lr_opt = max(utils.get_learning_rate(self.optimizer))
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = (param_group["lr"] / lr_opt) * self.args.lr
            except:
                self.logger.info('can not load state_dict to optimizer, so give up loading.')
        else:
            self.logger.info('do not load optimizer_state_dict.')

        self.logger.info(f"precision_test_best: {checkpoint['mAP_best']:.4f}, "
                         f"precision_test_curr: {checkpoint['mAP_curr']:.4f}, "
                         f"loss_test_best: {checkpoint['loss_val_best']:.4f}, "
                         # f"test_curr loss: {checkpoint['loss_val_curr']:.4f}, "
                         f"loss_train: {checkpoint['loss_tr']:.4f} in epoch {checkpoint['epoch']}, "
                         f"resuming from {file}.")
        for i in range(1, self.args.start_epoch):
            self.lr_scheduler.step()
        # self.lr_scheduler._step_count = self.args.start_epoch  # not working
            # for j in range(len(self.train_loader)):
        torch.cuda.empty_cache()

    def save_result(self, is_train, is_best=False):
        path = os.path.join(self.args.save_path, 'result_csv')
        # if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

        filename = f'results_L{self.args.lr:.1e}_train.csv' if is_train else f'results_L{self.args.lr:.1e}_test.csv'
        res_path = os.path.join(path, filename)

        print(f"score: {self.meter['ap'].scores[0].shape}-{len(self.meter['ap'].scores)}, "
              f"target: {self.meter['ap'].targets[0].shape}--{len(self.meter['ap'].targets)}, "
              f"name: {self.meter['ap'].filenames[0]}--{len(self.meter['ap'].filenames)}")

        with open(res_path, 'w') as fid:
            for i in range(self.meter['ap'].scores.shape[0]):
                fid.write('{},{},{}\n'.format(self.meter['ap'].filenames[i],
                                             ','.join(map(str, self.meter['ap'].scores[i].numpy())),
                                              ','.join(map(str, self.meter['ap'].targets[i].numpy()))))

        if is_best:
            filename_best = f'results_L{self.args.lr:.0e}_best.csv'  # the result of val predictions when the val set achieve best scores
            res_path_best = os.path.join(path, filename_best)
            shutil.copyfile(res_path, res_path_best)

    @staticmethod
    def convertDict_state(cpk):
        import collections
        cpk_ = collections.OrderedDict()
        for k, v in cpk.items():
            if k.startswith('module.'):
                cpk_[k[7:]] = v
        if len(cpk_) == 0:
            cpk_ = cpk
        return cpk_

    def reset_meters(self):
        self.meter['loss'] = metric.AverageMeter('loss')
        self.meter['loss_all'] = metric.AverageMeter('loss all rank')
        self.meter['ap'] = metric.AveragePrecisionMeter()


if __name__ == '__main__':
    import main as m
    from utilities import utils

    args = m.parser.parse_args()
    args.save_all_dir = ''
    utils.init(args)

    e = Engine(args)
    e.init()
    e.load_checkpoint()
