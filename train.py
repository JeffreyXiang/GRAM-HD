import os
import json
import time
import argparse

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from training_loop import training_process

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(i) for i in list(range(torch.cuda.device_count())))

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, opt):
    torch.manual_seed(rank)
    setup(rank, world_size, opt.port)
    device = torch.device(rank)
    training_process(rank, world_size, opt, device)
    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='results/GRAM64_FFHQ')
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='data/FFHQ')
    parser.add_argument('--config', type=str, default='GRAM64_FFHQ')
    parser.add_argument('--set_step', type=int, default=None, help='which step to load. None for latest')
    parser.add_argument('--total_step', type=int, default=100000, help='total training step')
    parser.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling")
    parser.add_argument('--save_interval', type=int, default=5000, help='interval between model saving')
    parser.add_argument('--port', type=str, default='12345')
    opt = parser.parse_args()

    os.makedirs(opt.output_dir, exist_ok=True)

    if opt.load_dir != '' and opt.set_step is None:
        opt.set_step = -1
        for filename in os.listdir(opt.load_dir):
            if 'step' in filename and 'pth' in filename:
                temp = int(filename[4:10])
                if temp > opt.set_step: opt.set_step = temp
        if opt.set_step < 0:
            opt.load_dir = ''
            opt.set_step = None
    print('='*80)
    print('Config')
    print('='*80)
    print(json.dumps(vars(opt), indent=4))
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    mp.spawn(train, args=(num_gpus, opt), nprocs=num_gpus, join=True)
