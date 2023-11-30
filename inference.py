import argparse
import numpy as np
import torch
import math
import os
from tqdm import tqdm
from generators import generators
import configs
import numpy as np
import imageio
import cv2


def parse_seeds(seeds):
    seeds = seeds.split(',')
    res = []
    for seed in seeds:
        if '-' in seed:
            start, end = seed.split('-')
            res += list(range(int(start), int(end)+1))
        else:
            res.append(int(seed))
    return res


def load_model(config, path):
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
    generator.load_state_dict(torch.load(path, map_location='cpu'))
    generator = generator.to('cuda')
    generator.eval()
    return generator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='./ckpts/ffhq1024.pt')
    parser.add_argument('--output_dir', type=str, default='./images')
    parser.add_argument('--config', type=str, default='GRAMHD1024_FFHQ')
    parser.add_argument('--seeds', type=str, default='0-9')
    parser.add_argument('--type', type=str, default='multiview1x3', choices=['video', 'multiview1x3', 'multiview3x7', 'rand'])
    parser.add_argument('--truncation', type=float, default=0.7)
    opt = parser.parse_args()

    os.makedirs(opt.output_dir, exist_ok=True)
    config = getattr(configs, opt.config)
    generator = load_model(config, opt.ckpt)
        
    # default setting
    if opt.type == 'video':
        h_mean = math.pi * (90 / 180)
        v_mean = math.pi * (90 / 180)
        generator.renderer.lock_view_dependence = True
        frames = 60
        concat = None
        face_yaws = list(np.linspace(-0.3, 0.3, frames // 2 + 1)[:-1]) + list(np.linspace(0.3, -0.3, frames // 2 + 1)[:-1])
        face_pitchs = [0*np.pi] * frames
        face_angles = [[a + v_mean, b + h_mean] for a, b in zip(face_pitchs, face_yaws)]
        fovs = [12] * frames
    elif opt.type == 'multiview1x3':
        h_mean = math.pi * (90 / 180)
        v_mean = math.pi * (90 / 180)
        generator.renderer.lock_view_dependence = True
        frames = 3
        concat = (1, 3)
        face_yaws = list(np.linspace(-0.3, 0.3, frames))
        face_pitchs = [0.0] * frames
        face_angles = [[a + v_mean, b + h_mean] for a, b in zip(face_pitchs, face_yaws)]
        fovs = [12] * frames
    elif opt.type == 'multiview3x7':
        h_mean = math.pi * (90 / 180)
        v_mean = math.pi * (90 / 180)
        generator.renderer.lock_view_dependence = True
        frames = 21
        concat = (3, 7)
        face_yaws = list(np.linspace(-0.4, 0.4, frames//3)) * 3
        face_pitchs = [0.2] * (frames // 3) + [0.0] * (frames // 3) + [-0.2] * (frames // 3)
        face_angles = [[a + v_mean, b + h_mean] for a, b in zip(face_pitchs, face_yaws)]
        fovs = [12] * frames
    elif opt.type == 'rand':
        generator.renderer.lock_view_dependence = False
        frames = 1
        concat = (1, 1)
        face_angles = [[None, None]]
        fovs = [12] * frames

    seeds = parse_seeds(opt.seeds)
    is_sr_model = isinstance(generator, generators.GramHDGenerator)
    for idx, seed in tqdm(enumerate(seeds), total=len(seeds), desc='Generating images'):
        images = np.zeros((frames, config['global']['img_size'], config['global']['img_size'], 3), dtype=np.uint8)
        if is_sr_model:
            lr_images = np.zeros((frames, config['global']['img_size'] // generator.scale_factor, config['global']['img_size'] // generator.scale_factor, 3), dtype=np.uint8)
        for i, ((pitch, yaw), fov) in enumerate(zip(face_angles, fovs)):
            config['camera']['fov'] = fov
            torch.manual_seed(seed)
            z = torch.randn((1, 256), device='cuda')
            with torch.no_grad():
                generator.get_avg_w()
                camera_origin = [np.sin(pitch) * np.cos(yaw), np.cos(pitch), np.sin(pitch) * np.sin(yaw)] if pitch is not None and yaw is not None else None
                img = generator(z, **config['camera'], camera_origin=camera_origin, truncation_psi=opt.truncation)[0]
                if is_sr_model:
                    lr_img = img[1]
                    img = img[0]
                    lr_img = lr_img * 0.5 + 0.5
                    lr_img = lr_img.permute(0, 2, 3, 1).squeeze().cpu().numpy()
                    lr_img = (lr_img * 255).astype(np.uint8)
                    lr_images[i] = np.nan_to_num(lr_img)
                img = img * 0.5 + 0.5
                img = img.permute(0, 2, 3, 1).squeeze().cpu().numpy()
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                images[i] = np.nan_to_num(img)
            
            if concat is None and opt.type != 'video':
                imageio.imsave(os.path.join(opt.output_dir, f'grid_{seed}_{i}.png'),images[i])                
                if is_sr_model: imageio.imsave(os.path.join(opt.output_dir, f'grid_{seed}_{i}_lr.png'),lr_images[i])

        if opt.type == 'video':
            imageio.mimsave(os.path.join(opt.output_dir, f'grid_{seed}.mp4'), images, fps=30)
            if is_sr_model: imageio.mimsave(os.path.join(opt.output_dir, f'grid_{seed}_lr.mp4'), lr_images, fps=30)
        elif opt.type == 'rand':
            imageio.imsave(os.path.join(opt.output_dir, f'{idx:05d}.png'), images[0])
        elif concat is not None:
            images = images.reshape((concat[0], concat[1], config['global']['img_size'], config['global']['img_size'], 3))
            images = np.concatenate(images, axis=-3)
            images = np.concatenate(images, axis=-2)
            imageio.imsave(os.path.join(opt.output_dir, f'grid_{seed}.png'), images)
            if is_sr_model:
                lr_images = lr_images.reshape((concat[0], concat[1], config['global']['img_size'] // generator.scale_factor, config['global']['img_size'] // generator.scale_factor, 3))
                lr_images = np.concatenate(lr_images, axis=-3)
                lr_images = np.concatenate(lr_images, axis=-2)
                imageio.imsave(os.path.join(opt.output_dir, f'grid_{seed}_lr.png'), lr_images)
