import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torch_utils.interpolate import *


def z_sampler(shape, device, dist):
    if dist == 'gaussian':
        z = torch.randn(shape, device=device)
    elif dist == 'uniform':
        z = torch.rand(shape, device=device) * 2 - 1
    return z


class Gan3DProcess:
    """
    Process for training 3D aware GANs (e.g. GRAM)

    Args:
        batch_split (int): Number of splits for batch.
        pos_lambda (float): Weight for position loss.
        real_pos_lambda (float): Weight for real position loss.
        r1_lambda (float): Weight for R1 gradient penalty.
        r1_interval (int): Interval for R1 gradient penalty.
    """

    def __init__(self, batch_split=1, pos_lambda=15., real_pos_lambda=1., r1_lambda=10., r1_interval=1) -> None:
        self.batch_split = batch_split
        self.pos_lambda = pos_lambda
        self.real_pos_lambda = real_pos_lambda
        self.r1_lambda = r1_lambda
        self.r1_interval = r1_interval
        self.fixed_z = z_sampler((25, 256), device='cpu', dist='gaussian')

    def train_D(self, real_imgs, real_positions, generator_ddp, discriminator_ddp, optimizer_D, scaler, config, device):
        with torch.cuda.amp.autocast():
            # Generate images for discriminator training
            with torch.no_grad():
                z = z_sampler((real_imgs.shape[0], generator_ddp.module.z_dim), device=device, dist='gaussian')
                split_batch_size = z.shape[0] // self.batch_split
                gen_imgs = []
                gen_positions = []
                for split in range(self.batch_split):
                    subset_z = z[split * split_batch_size:(split+1) * split_batch_size]
                    g_imgs, g_pos = generator_ddp(subset_z, **config['camera'])
                    gen_imgs.append(g_imgs)
                    gen_positions.append(g_pos)
                gen_imgs = torch.cat(gen_imgs, axis=0)
                gen_positions = torch.cat(gen_positions, axis=0)

            real_imgs.requires_grad = True
            r_preds, r_pred_position = discriminator_ddp(real_imgs)

        if self.r1_lambda > 0 and discriminator_ddp.module.step % self.r1_interval == 0:
            # Gradient penalty
            grad_real = torch.autograd.grad(outputs=scaler.scale(r_preds.sum()), inputs=real_imgs, create_graph=True)
            inv_scale = 1./scaler.get_scale()
            grad_real = [p * inv_scale for p in grad_real][0]

        with torch.cuda.amp.autocast():
            if self.r1_lambda > 0 and discriminator_ddp.module.step % self.r1_interval == 0:
                grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                grad_penalty = 0.5 * self.r1_lambda * grad_penalty
            else:
                grad_penalty = 0
            
            g_preds, g_pred_position = discriminator_ddp(gen_imgs.detach())
            position_penalty = nn.MSELoss()(g_pred_position, gen_positions) * self.pos_lambda
            if self.real_pos_lambda > 0:
                position_penalty += nn.MSELoss()(r_pred_position, real_positions) * self.real_pos_lambda
            identity_penalty = position_penalty
            d_loss = F.softplus(g_preds).mean() + F.softplus(-r_preds).mean() + grad_penalty + identity_penalty

        optimizer_D.zero_grad()
        scaler.scale(d_loss).backward()
        scaler.unscale_(optimizer_D)
        nn.utils.clip_grad_norm_(discriminator_ddp.parameters(), config['optimizer'].get('grad_clip', 0.3))
        scaler.step(optimizer_D)

        return d_loss.item()

    def train_G(self, real_imgs, generator_ddp, ema, ema2, discriminator_ddp, optimizer_G, scaler, config, device):
        z = z_sampler((real_imgs.shape[0], generator_ddp.module.z_dim), device=device, dist='gaussian')
        split_batch_size = z.shape[0] // self.batch_split

        for split in range(self.batch_split):
            with torch.cuda.amp.autocast():
                subset_z = z[split * split_batch_size:(split+1) * split_batch_size]
                gen_imgs, gen_positions = generator_ddp(subset_z, **config['camera'])
              
                g_preds, g_pred_position = discriminator_ddp(gen_imgs)
                position_penalty = nn.MSELoss()(g_pred_position, gen_positions) * self.pos_lambda
                identity_penalty = position_penalty
                g_loss = F.softplus(-g_preds).mean() + identity_penalty
                
            scaler.scale(g_loss).backward()

        scaler.unscale_(optimizer_G)
        nn.utils.clip_grad_norm_(generator_ddp.parameters(), config['optimizer'].get('grad_clip', 0.3))
        scaler.step(optimizer_G)
        scaler.update()
        optimizer_G.zero_grad()
        ema.update(generator_ddp.parameters())
        ema2.update(generator_ddp.parameters())

        return g_loss.item()

    def snapshot(self, generator_ddp, discriminator_ddp, config, output_dir, device, batchsize=1):
        with torch.no_grad():
            generator_ddp.module.get_avg_w()

            gen_imgs = []
            for i in range(0, 25, batchsize):
                gen_imgs.append(generator_ddp.module.forward(self.fixed_z[i:(i+batchsize)].to(device), **config['camera'], camera_origin=[0., 0., 1.], truncation_psi=0.7)[0])
            gen_imgs = torch.cat(gen_imgs, dim=0)
            save_image(gen_imgs[:25], os.path.join(output_dir, "%06d_fixed.png"%discriminator_ddp.module.step), nrow=5, normalize=True)

            gen_imgs = []
            for i in range(0, 25, batchsize):
                gen_imgs.append(generator_ddp.module.forward(self.fixed_z[i:(i+batchsize)].to(device), **config['camera'], camera_origin=[-np.sin(0.5), 0., np.cos(0.5)], truncation_psi=0.7)[0])
            gen_imgs = torch.cat(gen_imgs, dim=0)
            save_image(gen_imgs[:25], os.path.join(output_dir, "%06d_tilted.png"%discriminator_ddp.module.step), nrow=5, normalize=True)


class SRGan3DProcess:
    """
    Process for training 3D aware GANs with 3D super resolution (GRAM-HD)

    Args:
        batch_split (int): Number of splits for batch.
        pos_lambda (float): Weight for position loss.
        real_pos_lambda (float): Weight for real position loss.
        r1_lambda (float): Weight for R1 gradient penalty.
        r1_interval (int): Interval for R1 gradient penalty.
        cons_lambda (float): Weight for consistency loss.
        use_patch_d (bool): Whether to use patch discriminator.
        patch_lambda (float): Weight for patch discriminator.
        r1_patch (bool): Whether to use R1 gradient penalty for patch discriminator.
    """
    
    def __init__(self, batch_split=1, pos_lambda=15., real_pos_lambda=1., r1_lambda=10., r1_interval=1, cons_lambda=1, use_patch_d=False, patch_lambda=1., r1_patch=False) -> None:
        self.batch_split = batch_split
        self.pos_lambda = pos_lambda
        self.real_pos_lambda = real_pos_lambda
        self.r1_lambda = r1_lambda
        self.r1_interval = r1_interval
        self.cons_lambda = cons_lambda
        self.use_patch_d = use_patch_d
        self.patch_lambda = patch_lambda
        self.r1_patch = r1_patch
        self.fixed_z = z_sampler((25, 256), device='cpu', dist='gaussian')

    def train_D(self, real_imgs, real_positions, generator_ddp, discriminator_ddp, optimizer_D, scaler, config, device):
        real_imgs = real_imgs[:, :3, :, :]
       
        with torch.cuda.amp.autocast():
            # Generate images for discriminator training
            with torch.no_grad():
                z = z_sampler((real_imgs.shape[0], generator_ddp.module.z_dim), device=device, dist='gaussian')
                split_batch_size = z.shape[0] // self.batch_split
                gen_imgs = []
                gen_positions = []
                for split in range(self.batch_split):
                    subset_z = z[split * split_batch_size:(split+1) * split_batch_size]
                    (g_imgs, _, _, _), g_pos = generator_ddp(subset_z, **config['camera'])
                    gen_imgs.append(g_imgs)
                    gen_positions.append(g_pos)
                gen_imgs = torch.cat(gen_imgs, axis=0)
                gen_positions = torch.cat(gen_positions, axis=0)
            
            real_imgs.requires_grad = True
            if self.use_patch_d:
                r_preds, r_pred_position, r_patch_preds = discriminator_ddp(real_imgs)
            else:
                r_preds, r_pred_position = discriminator_ddp(real_imgs)
            
        if self.r1_lambda > 0 and discriminator_ddp.module.step % self.r1_interval == 0:
            # Gradient penalty
            if self.r1_patch:
                grad_real = torch.autograd.grad(outputs=scaler.scale(r_preds.sum() + self.patch_lambda * r_patch_preds.mean(-1).sum()), inputs=real_imgs, create_graph=True)
            else:
                grad_real = torch.autograd.grad(outputs=scaler.scale(r_preds.sum()), inputs=real_imgs, create_graph=True)
            inv_scale = 1./scaler.get_scale()
            grad_real = [p * inv_scale for p in grad_real][0]
        
        with torch.cuda.amp.autocast():
            if self.r1_lambda > 0 and discriminator_ddp.module.step % self.r1_interval == 0:
                grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                grad_penalty = 0.5 * self.r1_lambda * grad_penalty
            else:
                grad_penalty = 0
            
            if self.use_patch_d:
                g_preds, g_pred_position, g_patch_preds = discriminator_ddp(gen_imgs.detach())
            else:
                g_preds, g_pred_position = discriminator_ddp(gen_imgs.detach())
            position_penalty = nn.MSELoss()(g_pred_position, gen_positions) * self.pos_lambda
            if self.real_pos_lambda > 0:
                position_penalty += nn.MSELoss()(r_pred_position, real_positions) * self.real_pos_lambda
            identity_penalty = position_penalty

            patch_loss = 0
            if self.use_patch_d:
                patch_loss = self.patch_lambda * (F.softplus(g_patch_preds).mean() + F.softplus(-r_patch_preds).mean())
            main_loss = F.softplus(g_preds).mean() + F.softplus(-r_preds).mean()
            d_loss = main_loss + grad_penalty + identity_penalty + patch_loss

        optimizer_D.zero_grad()
        scaler.scale(d_loss).backward()
        scaler.unscale_(optimizer_D)
        nn.utils.clip_grad_norm_(discriminator_ddp.parameters(), config['optimizer'].get('grad_clip', 0.3))
        scaler.step(optimizer_D)

        return {'Total': d_loss.item(), 'Main': main_loss.item(), 'Patch': patch_loss.item(), 'R1': grad_penalty.item(), 'Pos': position_penalty.item()}

    def train_G(self, real_imgs, generator_ddp, ema, ema2, discriminator_ddp, optimizer_G, scaler, config, device):
        z = z_sampler((real_imgs.shape[0], generator_ddp.module.z_dim), device=device, dist='gaussian')
        split_batch_size = z.shape[0] // self.batch_split

        log = {'Total': 0.0, 'Main': 0.0, 'Patch': 0.0, 'Cons': 0.0, 'Pos': 0.0}

        for split in range(self.batch_split):
            with torch.cuda.amp.autocast():
                subset_z = z[split * split_batch_size:(split+1) * split_batch_size]
                (gen_imgs, lr_imgs, sr_rgba, lr_rgba), gen_positions = generator_ddp(subset_z, **config['camera'])
                
                if self.use_patch_d:
                    g_preds, g_pred_position, g_patch_preds = discriminator_ddp(gen_imgs)
                else:
                    g_preds, g_pred_position = discriminator_ddp(gen_imgs)
                position_penalty = nn.MSELoss()(g_pred_position, gen_positions) * self.pos_lambda
                identity_penalty = position_penalty

                cons_penalty = self.cons_lambda * ((bicubic_downsample(gen_imgs, generator_ddp.module.scale_factor) - lr_imgs)**2).mean()
                cons_penalty += self.cons_lambda * ((bicubic_downsample(sr_rgba, generator_ddp.module.scale_factor) - lr_rgba)**2).mean()
              
                patch_loss = 0
                if self.use_patch_d:
                    patch_loss = self.patch_lambda * F.softplus(-g_patch_preds).mean()
                main_loss = F.softplus(-g_preds).mean()
                g_loss = main_loss + identity_penalty + cons_penalty + patch_loss

            g_loss = g_loss / self.batch_split
            scaler.scale(g_loss).backward()
            log['Total'] += g_loss.item()
            log['Main'] += main_loss.item() / self.batch_split
            log['Patch'] += patch_loss.item() / self.batch_split
            log['Cons'] += cons_penalty.item() / self.batch_split
            log['Pos'] += position_penalty.item() / self.batch_split

        scaler.unscale_(optimizer_G)
        nn.utils.clip_grad_norm_(generator_ddp.parameters(), config['optimizer'].get('grad_clip', 0.3))
        scaler.step(optimizer_G)
        scaler.update()
        optimizer_G.zero_grad()
        ema.update(generator_ddp.parameters())
        ema2.update(generator_ddp.parameters())

        return log

    def snapshot(self, generator_ddp, discriminator_ddp, config, output_dir, device, batchsize=1):
        with torch.no_grad():
            generator_ddp.module.get_avg_w()

            gen_imgs = []
            sr_gen_imgs = []
            for i in range(0, 25, batchsize):
                imgs = generator_ddp.module.forward(self.fixed_z[i:(i+batchsize)].to(device), **config['camera'], camera_origin=[0., 0., 1.], truncation_psi=0.7)[0]
                gen_imgs.append(imgs[1])
                sr_gen_imgs.append(imgs[0])
            gen_imgs = torch.cat(gen_imgs, dim=0)
            sr_gen_imgs = torch.cat(sr_gen_imgs, dim=0)
            save_image(gen_imgs[:25], os.path.join(output_dir, "%06d_fixed.png"%discriminator_ddp.module.step), nrow=5, normalize=True, range=(-1, 1))
            save_image(sr_gen_imgs[:25], os.path.join(output_dir, "%06d_fixed_sr.png"%discriminator_ddp.module.step), nrow=5, normalize=True, range=(-1, 1))

            gen_imgs = []
            sr_gen_imgs = []
            for i in range(0, 25, batchsize):
                imgs = generator_ddp.module.forward(self.fixed_z[i:(i+batchsize)].to(device), **config['camera'], camera_origin=[-np.sin(0.5), 0., np.cos(0.5)], truncation_psi=0.7)[0]
                gen_imgs.append(imgs[1])
                sr_gen_imgs.append(imgs[0])
            gen_imgs = torch.cat(gen_imgs, dim=0)
            sr_gen_imgs = torch.cat(sr_gen_imgs, dim=0)
            save_image(gen_imgs[:25], os.path.join(output_dir, "%06d_tilted.png"%discriminator_ddp.module.step), nrow=5, normalize=True, range=(-1, 1))
            save_image(sr_gen_imgs[:25], os.path.join(output_dir, "%06d_tilted_sr.png"%discriminator_ddp.module.step), nrow=5, normalize=True, range=(-1, 1))
