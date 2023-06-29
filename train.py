import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
import random
import argparse
import torch
import numpy as np

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter
from torch.cuda import amp

from utils.utils import get_loader

def parser():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--data_path", help='path to image folder')
    
    # label
    parser.add_argument("--three_corner", default='cfgs/three_corner.json', help='path to three_corner.json')
    parser.add_argument("--four_corner", default='cfgs/four_corner.json', help='path to four_corner.json')
    parser.add_argument("--stroke", default='cfgs/stroke_count.json', help='path to stroke_count.json')
    # parser.add_argument("--stroke", default='cfgs/stroke_frequency.json', help='path to stroke_frequency.json')
    
    # setting
    parser.add_argument("--seed", default=0, type=int, help='init seed')
    parser.add_argument("--batch_size", default=64, type=int, help="batch size while training")
    parser.add_argument("--num_workers", default=6, type=int, help="number of dataloader")
    
    # ddp
    parser.add_argument("--ddp", default=True, type=bool, help="use ddp or not")
    parser.add_argument("--port", default=8888, type=int, help='ddp port')
    
    # parser.add_argument("", help="")

    args = parser.parse_args()

    return args

def init(seed):
    seed = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

def cleanup():
    dist.destroy_process_group()

def is_main_worker(gpu):
    return (gpu <= 0)

def train_ddp(rank, world_size, cfgs):

    port = cfgs.port
    dist.init_process_group(
        backend='nccl',
        init_method="tcp://127.0.0.1:" + str(port),
        world_size=world_size,
        rank=rank,
    )

    train(cfgs, ddp_gpu=rank)
    cleanup()

def train(cfgs, ddp_gpu=-1):
    # set gpu of each multiprocessing
    torch.cuda.set_device(ddp_gpu)
    
    # get dataLoader
    train_loader, val_loader = get_loader(cfgs.data_path, cfgs.ddp, cfgs.batch_size, cfgs.num_workers) 
    print("Get data loader successfully")
    


if __name__ == '__main__':
    cfgs = parser()
    if cfgs.ddp:
        n_gpus_per_node = torch.cuda.device_count()
        world_size = n_gpus_per_node
        mp.spawn(train_ddp, nprocs=n_gpus_per_node, args=(world_size, cfgs))
    else: 
        train()