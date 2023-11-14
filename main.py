import torch
import torch.distributed as dist
import argparse
from utilities import utils
from utilities import utils_ddp
from engine import Engine

parser = argparse.ArgumentParser(description="Model Configuration")
parser.add_argument('-m', '--model', default='ResNet-101', type=str, help="The model to be ran according to ./models package")
parser.add_argument('-lr', '--lr', '--learning-rate', default=0.05, type=float, help="Learning rate")
parser.add_argument('-bs', '--batch-size', default=32, type=int, help="Batch size for training and valuating")
parser.add_argument('-ds', '--data-set', default='MS-COCO', type=str, help="Data set [MS-COCO, VOC2007]")
parser.add_argument('-dr', '--data-root', default=r'/pcalab/tmp/Lanke/Data_set', type=str, help="Your Data Set Root")
parser.add_argument('-cd', '--cuda_devices', default=[0, 1], nargs='+', type=int, help="Cuda device ids for running")
parser.add_argument('-mixer', default='', type=str, help="'SpliceMix--Default=True' for the default setting; 'SpliceMix--Mini=True' for minimalism setting")
parser.add_argument('-grids', default=['2x2', ], nargs='+', type=str, help="Grid strategy, e.g., ['1x2', '2x2-2', ...]")
parser.add_argument('-n_grids', default=[0, ], nargs='+', type=int, help="Number of mixed images in each strategy; 0 denotes a quarter of regualr batch size will be used.")

# Secondary Parameters
parser.add_argument('-ims', '--image-size', default=448, type=int, help="Image size for training and testing")
parser.add_argument('-eps', '--epoch-step', default=[40, 60], nargs='+', type=int, help="Epoch step for linearly decaying learning rate")
parser.add_argument('-ep', '--epochs', default=80, type=int, help="Max epochs")
parser.add_argument('-o', '--optimizer', default='SGD', type=str, help="The optimizer can be only chosen from {\'SGD\', \'Adam\'} for now. More may be implemented later")
parser.add_argument('-e', '--evaluate', default=1, type=int, help="-1: do not evaluate; 0: enter evaluation mode; e>0: evaluate once per e times of training epoch")
parser.add_argument('-lrp', '--lrp', '--learning-rate-pretrained', default=0.1, type=float, help="The learning rate decay for pretrained layers")
parser.add_argument('-r', '--resume', default='', type=str, help="Checkpoint path for resuming training")
parser.add_argument('-loadOpt', '--load-optimizer', default=False, action='store_true', help="Whether load optimizer state_dcit when resuming training")
parser.add_argument('-sep', '--start-epoch', default=0, type=int, help="epoch to start")
parser.add_argument('-motum', '--momentum', default=0.9, type=float, help="Momentum for SGD")  # For Adam optimizer, the default parameters are used.
parser.add_argument('-wd', '--weight-decay', default=1e-4, type=float, help="Weight decay for SGD or Adam.")
parser.add_argument('-lrd', '--lr_decay','--learning-rate-decay', default=0.1, type=float, help="Learning rate decay for each 'epoch step'")
parser.add_argument('--seed', default=95, type=int, help="Seed for random, numpy.random, torch.random, torch.cuda")
parser.add_argument('--save-dir', default=r'checkpoint', type=str, help="The save directory for selected checkpoints of model")
parser.add_argument('-ver', '--print-verbose', default=1, type=int, help="0: no verbose; 1: print CP, CR, CF1, OP, OR, OF1; 2: print 1 + top_k")
parser.add_argument('--acc-top-k', default=3, type=int, help="Top k metrics")
parser.add_argument('-wup', '--warmup-epochs', default=0, type=int, help="0: not use warmup; >1: epochs for warmup")
parser.add_argument('-dis_amp', '--disable-amp', default=False, action='store_true', help="True: not use amp (Automatic Mixed Precision); False: use amp")
parser.add_argument('--backend', default='nccl', type=str, help="")
parser.add_argument('--master-addr', default='127.0.0.1', type=str, help="")
parser.add_argument('-P', '--master-port', default='17837', type=str, help="")
parser.add_argument('--local_rank', type=int, help="")
parser.add_argument('-Sprob', default=1., type=float, help="thersold for using SpliceMix")
parser.add_argument('-rmk', '--remark', default='', type=str, help="etc")


def main(rank, args):

    dist.init_process_group(args.backend, rank=rank, world_size=args.world_size)
    print(f'finish init, rank: {args.local_rank}, world size: {args.world_size}\t')

    torch.cuda.set_device(rank)
    utils_ddp.setup_for_distributed(rank==0)  # print master process only
    engine = Engine(args)
    if args.evaluate > 0:
        engine.train()
    else:
        engine.evaluate()
    print('Finish processes.')
    utils_ddp.cleanup()

if __name__ == "__main__":
    args = parser.parse_args()
    args = utils.init(args)
    main(args.local_rank, args)



