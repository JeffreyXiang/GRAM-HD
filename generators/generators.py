import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .representations.gram import *
from .super_resolutions.esrgan import *
from .super_resolutions.styleesrgan import *
from .renderers.manifold_renderer import *
from .renderers.manifold_sr_renderer import *


def sample_camera_positions(device, n=1, r=1, horizontal_stddev=1, vertical_stddev=1, horizontal_mean=math.pi*0.5, vertical_mean=math.pi*0.5, mode='normal'):
    """Samples n random locations along a sphere of radius r. Uses a gaussian distribution for pitch and yaw"""
    if mode == 'uniform':
        theta = (torch.rand((n, 1), device=device) - 0.5) * 2 * horizontal_stddev + horizontal_mean
        phi = (torch.rand((n, 1), device=device) - 0.5) * 2 * vertical_stddev + vertical_mean

    elif mode == 'normal' or mode == 'gaussian':
        theta = torch.randn((n, 1), device=device) * horizontal_stddev + horizontal_mean
        phi = torch.randn((n, 1), device=device) * vertical_stddev + vertical_mean
    elif mode == 'hybrid':
        if random.random() < 0.5:
            theta = (torch.rand((n, 1), device=device) - 0.5) * 2 * horizontal_stddev * 2 + horizontal_mean
            phi = (torch.rand((n, 1), device=device) - 0.5) * 2 * vertical_stddev * 2 + vertical_mean
        else:
            theta = torch.randn((n, 1), device=device) * horizontal_stddev + horizontal_mean
            phi = torch.randn((n, 1), device=device) * vertical_stddev + vertical_mean
    elif mode == 'spherical_uniform':
        theta = (torch.rand((n, 1), device=device) - .5) * 2 * horizontal_stddev + horizontal_mean
        v_stddev, v_mean = vertical_stddev / math.pi, vertical_mean / math.pi # convert from radians to [0,1]
        v = ((torch.rand((n,1), device=device) - .5) * 2 * v_stddev + v_mean)
        v = torch.clamp(v, 1e-5, 1 - 1e-5)
        phi = torch.arccos(1 - 2 * v)
    elif mode == 'cuhk':
        view_count_list = [97,    0,    0,    0,    0,    0,    0,    0,    0,    0,   10, 
        1,    0,    0,    2,   73,   16,    3,    1,    0,  390,   44,
        5,    1,    0, 2681,   99,    1,    0,    0, 2277,   71,    0,
        0,    0,  276,    7,    0,    0,    0,   58,    1,    0,    0,
        0,  200,    6,    0,    0,    0,  231,    8,    0,    0,    0,
        2332,   61,    0,    0,    0, 2261,   97,    1,    0,    0,  403,
        33,    5,    0,    0,   84,   11,    3,    1,    0,   22,    2,
        1,    0,    0,    2,    0,    0,    0,    0,  110,    0,    0,
        0,    0,    0,    0,    0,    0,    0,    1,    0,    0,    0,
        0,    1,    0,    3,    7,    6,   83,    0,    0,    2,    0,
        530,    1,    1,    0,    0, 2594,    5,    1,    0,    0, 1493,
        3,    0,    1,    0,  139,    0,    0,    0,    0,   38,   77,
        0,    0,    0,   14,    4,    1,    0,    0,   96,    0,    0,
        0,    0, 1237,    0,    0,    1,    0, 2295,   16,    0,    0,
        0,  557,    0,    0,    0,    0,  100,    0,    0,    0,    0,
        4,    0,    2,    2,    0,    7,    0,    0,    0,    0,   12,
        0,    0,    0,    0]

        count_sum = sum(view_count_list)
        view_bin_prob = [count / count_sum for count in view_count_list]
        label_sampled = np.random.choice(a=[i for i in range(len(view_count_list))], size=n, p=view_bin_prob)
        n_azi_id = label_sampled // 5
        n_ele_id = label_sampled - n_azi_id * 5
        batch_view = [[random.uniform(n_azi_id[k] * 10, n_azi_id[k] * 10 + 10) - 90, random.uniform(n_ele_id[k] * 10, n_ele_id[k] * 10 + 10)] for k in range(n)]
        batch_view = np.array(batch_view, dtype=np.float32)
        batch_view = batch_view * math.pi / 180.0

        theta = torch.from_numpy(batch_view[:,:1]).to(device) + math.pi*0.5
        phi = torch.from_numpy(batch_view[:,1:]).to(device) + math.pi*0.5

    else:
        theta = torch.ones((n, 1), device=device, dtype=torch.float) * horizontal_mean
        phi = torch.ones((n, 1), device=device, dtype=torch.float) * vertical_mean

    phi = torch.clamp(phi, 1e-5, math.pi - 1e-5)

    camera_origin = torch.zeros((n, 3), device=device)# torch.cuda.FloatTensor(n, 3).fill_(0)#torch.zeros((n, 3))

    camera_origin[:, 0:1] = r*torch.sin(phi) * torch.cos(theta)
    camera_origin[:, 2:3] = r*torch.sin(phi) * torch.sin(theta)
    camera_origin[:, 1:2] = r*torch.cos(phi)

    return camera_origin, torch.cat([phi, theta], dim=-1)


class Generator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.epoch = 0
        self.step = 0


class GramGenerator(Generator):
    """
    GRAM Generator

    Args:
        z_dim (int): Dimension of latent space
        img_size (int): Size of image
        h_stddev (float): Camera horizontal standard deviation
        v_stddev (float): Camera vertical standard deviation
        h_mean (float): Camera horizontal mean
        v_mean (float): Camera vertical mean
        sample_dist (str): Camera sampling distribution
        representation_kwargs (dict): Arguments for representation
        renderer_kwargs (dict): Arguments for renderer
    """
    def __init__(self, z_dim, img_size, h_stddev, v_stddev, h_mean, v_mean, sample_dist, representation_kwargs, renderer_kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.img_size = img_size
        self.h_stddev = h_stddev
        self.v_stddev = v_stddev
        self.h_mean = h_mean
        self.v_mean = v_mean
        self.sample_dist = sample_dist
        self.representation = Gram(z_dim, **representation_kwargs)
        self.renderer = ManifoldRenderer(**renderer_kwargs)

    def _volume(self, z, truncation_psi=1):
        return lambda points, ray_directions: self.representation.get_radiance(z, points, ray_directions, truncation_psi)

    def _intersections(self, points, levels):
        return self.representation.get_intersections(points, levels)

    def get_avg_w(self):
        self.representation.get_avg_w()

    def forward(self, z, fov, ray_start, ray_end, img_size=None, camera_origin=None, camera_pos=None, truncation_psi=1):
        if camera_origin is None:
            camera_origin, camera_pos = sample_camera_positions(z.device, z.shape[0], 1, self.h_stddev, self.v_stddev, self.h_mean, self.v_mean, self.sample_dist)
        else:
            camera_origin = torch.tensor(camera_origin, dtype=torch.float32, device=z.device).reshape(1, 3).expand(z.shape[0], 3)
        if img_size is None:
            img_size = self.img_size
        img, _ = self.renderer.render(self._intersections, self._volume(z, truncation_psi), img_size, camera_origin, camera_pos, fov, ray_start, ray_end, z.device)
        return img, camera_pos

  
class GramHDGenerator(Generator):
    """
    GRAM-HD Generator

    Args:
        z_dim (int): Dimension of latent space
        feature_dim (int): Dimension of feature space
        img_size (int): Size of image
        h_stddev (float): Camera horizontal standard deviation
        v_stddev (float): Camera vertical standard deviation
        h_mean (float): Camera horizontal mean
        v_mean (float): Camera vertical mean
        sample_dist (str): Camera sampling distribution
        gram_model_file (str): Path to pretrained GRAM model
        representation_kwargs (dict): Arguments for representation
        super_resolution_kwargs (dict): Arguments for SR module
        renderer_kwargs (dict): Arguments for renderer
        lr_img_size (int): Size of low resolution image
    """
    def __init__(self, z_dim, feature_dim, img_size, h_stddev, v_stddev, h_mean, v_mean, sample_dist, representation_kwargs, super_resolution_kwargs, renderer_kwargs, gram_model_file=None, lr_img_size=None):
        super().__init__()
        self.z_dim = z_dim
        self.feature_dim = feature_dim
        if lr_img_size is None:
            self.scale_factor = 4
        else:
            self.scale_factor = img_size // lr_img_size
        self.img_size = img_size
        self.h_stddev = h_stddev
        self.v_stddev = v_stddev
        self.h_mean = h_mean
        self.v_mean = v_mean
        self.sample_dist = sample_dist
        self.representation = Gram(z_dim, feature_dim, **representation_kwargs)
        self.super_resolution = StyleRRDBNet(4 + feature_dim, 4, scale_factor = self.scale_factor, use_mapping_network=True, **super_resolution_kwargs['fg'])
        self.super_resolution_bg = RRDBNet(3, 3, scale_factor = self.scale_factor, **super_resolution_kwargs['bg'])
        self.renderer = ManifoldSRRenderer(scale_factor = self.scale_factor, **renderer_kwargs)

        # load pretrained GRAM model
        if gram_model_file is not None:
            ema = torch.load(gram_model_file.replace('generator', 'ema'), map_location='cpu')
            parameters = [p for p in self.representation.pretrained_params() if p.requires_grad]
            ema.copy_to(parameters)
        self.representation.freeze_pretrained_params()

    def _volume(self, frequencies, phase_shifts, truncation_psi=1):
        if truncation_psi < 1:
            frequencies = self.representation.rf_network.avg_frequencies.lerp(frequencies, truncation_psi)
            phase_shifts = self.representation.rf_network.avg_phase_shifts.lerp(phase_shifts, truncation_psi)
        return lambda points, ray_directions: self.representation.rf_network.forward_with_frequencies_phase_shifts(points, frequencies, phase_shifts, ray_directions)

    def _feature(self, frequencies, phase_shifts, truncation_psi=1):
        if truncation_psi < 1:
            frequencies = self.representation.rf_network.avg_frequencies.lerp(frequencies, truncation_psi)
            phase_shifts = self.representation.rf_network.avg_phase_shifts.lerp(phase_shifts, truncation_psi)
        return lambda points, ray_directions: self.representation.rf_network.forward_feature_with_frequencies_phase_shifts(points, frequencies, phase_shifts, ray_directions)

    def _intersections(self, points, levels):
        return self.representation.get_intersections(points, levels)

    def _super_resolution(self, z, truncation_psi=1):
        return lambda x: self.super_resolution(x, z, truncation_psi)

    def get_avg_w(self):
        z, ws = self.representation.get_avg_w()[2]
        self.super_resolution.get_avg_w(z)

    def forward(self, z, fov, ray_start, ray_end, img_size=None, camera_origin=None, camera_pos=None, truncation_psi=1):
        if camera_origin is None:
            camera_origin, camera_pos = sample_camera_positions(z.device, z.shape[0], 1, self.h_stddev, self.v_stddev, self.h_mean, self.v_mean, self.sample_dist)
        else:
            camera_origin = torch.tensor(camera_origin, dtype=torch.float32, device=z.device).reshape(1, 3).expand(z.shape[0], 3)
        if img_size is None:
            img_size = self.img_size
        frequencies, phase_shifts = self.representation.rf_network.mapping_network(z)
        img, _ = self.renderer.render(self._intersections, self._volume(frequencies, phase_shifts, truncation_psi), self._feature(frequencies, phase_shifts, truncation_psi), self._super_resolution(z, truncation_psi), self.super_resolution_bg, img_size, camera_origin, camera_pos, fov, ray_start, ray_end, z.device)
        return img, camera_pos
