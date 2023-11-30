
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
from torch_ema import ExponentialMovingAverage

from generators import generators
from discriminators import discriminators
from processes import processes
import configs as configs
import datasets


def set_generator(config, device, opt):
    generator_args = {}
    if 'representation' in config['generator']:
        generator_args['representation_kwargs'] = config['generator']['representation']['kwargs']
    if 'super_resolution' in config['generator']:
        generator_args['super_resolution_kwargs'] = config['generator']['super_resolution']['kwargs']
    if 'renderer' in config['generator']:
        generator_args['renderer_kwargs'] = config['generator']['renderer']['kwargs']
    generator = getattr(generators, config['generator']['class'])(
        **generator_args,
        **config['generator']['kwargs']
    )

    if opt.load_dir != '':
        generator.load_state_dict(torch.load(os.path.join(opt.load_dir, 'step%06d_generator.pth'%opt.set_step), map_location='cpu'))

    generator = generator.to(device)
    
    if opt.load_dir != '':
        ema = torch.load(os.path.join(opt.load_dir, 'step%06d_ema.pth'%opt.set_step), map_location=device)
        ema2 = torch.load(os.path.join(opt.load_dir, 'step%06d_ema2.pth'%opt.set_step), map_location=device)
        parameters_ = [p for p in generator.parameters()]
        if len(parameters_) == len(ema.shadow_params):
            for i in range(len(parameters_) - 1, -1, -1):
                if not parameters_[i].requires_grad:
                    ema.shadow_params.pop(i)
                    ema2.shadow_params.pop(i)
    else:
        ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
        ema2 = ExponentialMovingAverage(generator.parameters(), decay=0.9999)
    
    return generator, ema, ema2


def set_discriminator(config, device, opt):
    discriminator = getattr(discriminators, config['discriminator']['class'])(**config['discriminator']['kwargs'])

    if opt.load_dir != '':
        discriminator.load_state_dict(torch.load(os.path.join(opt.load_dir, 'step%06d_discriminator.pth'%opt.set_step), map_location='cpu'))
    
    discriminator = discriminator.to(device)
    return discriminator


def set_optimizer_G(generator_ddp, config, opt):
    param_groups = []
    if 'sr_lr' in config['optimizer']:
        sr_parameters = [p for n, p in generator_ddp.named_parameters() if 'module.super_resolution' in n]
        param_groups.append({'params': sr_parameters, 'name': 'sr_parameters', 'lr':config['optimizer']['sr_lr']})
    if 'sr_mapping_lr' in config['optimizer']:
        sr_mapping_parameters = [p for n, p in generator_ddp.named_parameters() if 'module.super_resolution.mapping_network' in n]
        param_groups.append({'params': sr_mapping_parameters, 'name': 'sr_mapping_parameters', 'lr':config['optimizer']['sr_mapping_lr']})
    generator_parameters = [p for n, p in generator_ddp.named_parameters() if 
        ('sr_lr' not in config['optimizer'] or 'module.super_resolution' not in n) and
        ('sr_mapping_lr' not in config['optimizer'] or 'module.super_resolution.mapping_network' not in n)
    ]
    param_groups.append({'params': generator_parameters, 'name': 'generator'})
    optimizer_G = torch.optim.Adam(param_groups, lr=config['optimizer']['gen_lr'], betas=config['optimizer']['betas'])

    if opt.load_dir != '':
        state_dict = torch.load(os.path.join(opt.load_dir, 'step%06d_optimizer_G.pth'%opt.set_step), map_location='cpu')
        optimizer_G.load_state_dict(state_dict)
    
    return optimizer_G


def set_optimizer_D(discriminator_ddp, config, opt):
    optimizer_D = torch.optim.Adam(discriminator_ddp.parameters(), lr=config['optimizer']['disc_lr'], betas=config['optimizer']['betas'])

    if opt.load_dir != '':
        optimizer_D.load_state_dict(torch.load(os.path.join(opt.load_dir, 'step%06d_optimizer_D.pth'%opt.set_step), map_location='cpu'))
    
    return optimizer_D


def training_process(rank, world_size, opt, device):

#--------------------------------------------------------------------------------------
# extract training config
    config = getattr(configs, opt.config)

#--------------------------------------------------------------------------------------
# set amp gradient scaler
    scaler = torch.cuda.amp.GradScaler()
    if opt.load_dir != '':
        if not config['global'].get('disable_scaler', False):
            scaler.load_state_dict(torch.load(os.path.join(opt.load_dir, 'step%06d_scaler.pth'%opt.set_step)))

    if config['global'].get('disable_scaler', False):
        scaler = torch.cuda.amp.GradScaler(enabled=False)

#--------------------------------------------------------------------------------------
#set generator and discriminator
    generator, ema, ema2 = set_generator(config, device, opt)
    discriminator = set_discriminator(config, device, opt)

    generator_ddp = DDP(generator, device_ids=[rank], find_unused_parameters=False)
    discriminator_ddp = DDP(discriminator, device_ids=[rank], find_unused_parameters=False, broadcast_buffers=False)

    generator = generator_ddp.module
    discriminator = discriminator_ddp.module

    if rank == 0:
        print('\n' + '='*80)
        print('Model Summary')
        print('='*80)
        for name, param in generator_ddp.named_parameters():
            print(f'{name:<{96}}{param.shape}')
        total_num = sum(p.numel() for p in generator_ddp.parameters())
        trainable_num = sum(p.numel() for p in generator_ddp.parameters() if p.requires_grad)
        print('G: Total ', total_num, ' Trainable ', trainable_num)
        
        for name, param in discriminator_ddp.named_parameters():
            print(f'{name:<{96}}{param.shape}')
        total_num = sum(p.numel() for p in discriminator_ddp.parameters())
        trainable_num = sum(p.numel() for p in discriminator_ddp.parameters() if p.requires_grad)
        print('D: Total ', total_num, ' Trainable ', trainable_num)

#--------------------------------------------------------------------------------------
# set optimizers
    optimizer_G = set_optimizer_G(generator_ddp, config, opt)
    optimizer_D = set_optimizer_D(discriminator_ddp, config, opt)

    generator_losses = []
    discriminator_losses = []

    if opt.set_step != None:
        generator.step = opt.set_step
        discriminator.step = opt.set_step

#--------------------------------------------------------------------------------------
# set loss
    process = getattr(processes, config['process']['class'])(**config['process']['kwargs'])

#--------------------------------------------------------------------------------------
# get dataset
    dataset = getattr(datasets, config['dataset']['class'])(opt.data_dir, **config['dataset']['kwargs'])
    dataloader, CHANNELS = datasets.get_dataset_distributed_(dataset,
                    world_size,
                    rank,
                    config['global']['batch_size'])

#--------------------------------------------------------------------------------------
# main training loop
    with open(os.path.join(opt.output_dir, 'options.txt'), 'w') as f:
        f.write(str(opt))
        f.write('\n\n')
        f.write(str(generator))
        f.write('\n\n')
        f.write(str(discriminator))
        f.write('\n\n')
        f.write(str(opt.config))
        f.write('\n\n')
        f.write(str(config))

    with tqdm(desc="Steps", total=opt.total_step, initial=generator.step, disable=(rank!=0)) as pbar:
        while True:             
            #--------------------------------------------------------------------------------------
            # trainging iterations
            for i, (imgs, poses) in enumerate(dataloader):
                if scaler.get_scale() < 1:
                    scaler.update(1.)

                real_imgs = imgs.to(device, non_blocking=True)
                real_poses = poses.to(device, non_blocking=True)
                
                #--------------------------------------------------------------------------------------
                # TRAIN DISCRIMINATOR
                d_loss = process.train_D(real_imgs, real_poses, generator_ddp, discriminator_ddp, optimizer_D, scaler, config, device)
                discriminator_losses.append(d_loss)

                # TRAIN GENERATOR
                g_loss = process.train_G(real_imgs, generator_ddp, ema, ema2, discriminator_ddp, optimizer_G, scaler, config, device)
                generator_losses.append(g_loss)

                pbar.update(1)
                discriminator.step += 1
                generator.step += 1

                #--------------------------------------------------------------------------------------
                # print and save
                if rank == 0:
                    if i%10 == 0:
                        tqdm.write(f"[Experiment: {opt.output_dir}] [GPU: {os.environ['CUDA_VISIBLE_DEVICES']}] [Step: {discriminator.step}] [D loss: {d_loss}] [G loss: {g_loss}] [Scale: {scaler.get_scale()}]")

                    # save fixed angle generated images
                    if discriminator.step % opt.sample_interval == 0:
                        process.snapshot(generator_ddp, discriminator_ddp, config, opt.output_dir, device)

                    # save_model
                    if discriminator.step % opt.save_interval == 0:
                        torch.save(ema, os.path.join(opt.output_dir, 'step%06d_ema.pth'%discriminator.step))
                        torch.save(ema2, os.path.join(opt.output_dir, 'step%06d_ema2.pth'%discriminator.step))
                        torch.save(generator_ddp.module.state_dict(), os.path.join(opt.output_dir, 'step%06d_generator.pth'%discriminator.step))
                        torch.save(discriminator_ddp.module.state_dict(), os.path.join(opt.output_dir, 'step%06d_discriminator.pth'%discriminator.step))
                        torch.save(optimizer_G.state_dict(), os.path.join(opt.output_dir, 'step%06d_optimizer_G.pth'%discriminator.step))
                        torch.save(optimizer_D.state_dict(), os.path.join(opt.output_dir, 'step%06d_optimizer_D.pth'%discriminator.step))
                        torch.save(scaler.state_dict(), os.path.join(opt.output_dir, 'step%06d_scaler.pth'%discriminator.step))
                        torch.save(generator_losses, os.path.join(opt.output_dir, 'generator.losses'))
                        torch.save(discriminator_losses, os.path.join(opt.output_dir, 'discriminator.losses'))                                
                #--------------------------------------------------------------------------------------
