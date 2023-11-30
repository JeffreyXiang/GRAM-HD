import math
import torch
from torchvision.transforms import GaussianBlur

"""=============== BILINEAR ==============="""

bilinear_kernel = [None]

def bilinear_function(x):
    x = abs(x)
    res = 1 - x
    return res


def generate_bilinear_kernel(factor: int):
    size = 2 * factor
    func = torch.zeros(size, 1)
    for i in range(size):
        func[i] = bilinear_function((i - size / 2 + 0.5) / factor) / factor

    kernel = func @ func.t()

    return kernel


def bilinear_kernel_init(num):
    global bilinear_kernel
    for i in range(1, num + 1):
        kernel2d = generate_bilinear_kernel(i)
        bilinear_kernel.append(kernel2d)


def bilinear_downsample(input, factor: int):
    channels = input.shape[1]
    kernel = torch.zeros(channels, channels, 2 * factor, 2 * factor)
    for i in range(channels):
        kernel[i, i] = bilinear_kernel[factor]
    input = torch.nn.functional.pad(input, [int(0.5 * factor)] * 4, 'replicate')
    res = torch.nn.functional.conv2d(input, kernel.to(input.device), stride=factor)

    return res

"""=============== BICUBIC ==============="""

bicubic_kernel = [None]

def bicubic_function(x):
    x = abs(x)
    if x <= 1:
        res = 1.5 * x**3 - 2.5 * x**2 + 1
    elif x < 2:
        res = -0.5 * x**3 + 2.5 * x**2 - 4 * x + 2
    else:
        res = 0

    return res


def generate_bicubic_kernel(factor: int):
    size = 4 * factor
    func = torch.zeros(size, 1)
    for i in range(size):
        func[i] = bicubic_function((i - size / 2 + 0.5) / factor) / factor

    kernel = func @ func.t()

    return kernel


def bicubic_kernel_init(num):
    global bicubic_kernel
    for i in range(1,num + 1):
        kernel2d = generate_bicubic_kernel(i)
        bicubic_kernel.append(kernel2d)


def bicubic_downsample(input, factor:int):
    channels = input.shape[1]
    kernel = torch.zeros(channels, channels, 4 * factor, 4 * factor)
    for i in range(channels):
        kernel[i, i] = bicubic_kernel[factor]
    input = torch.nn.functional.pad(input, [int(1.5 * factor)] * 4, 'replicate')
    res = torch.nn.functional.conv2d(input, kernel.to(input.device), stride=factor)
    
    return res

"""=============== GAUSSIAN ==============="""

def generate_gaussian_kernel(ksize, sigma):
    func = torch.linspace(-(ksize - 1) / 2, (ksize - 1) / 2, ksize).reshape(ksize, 1)
    func = torch.exp(-(func * func) / (2 * sigma * sigma))
    func = func / func.sum()
    return func

def gaussian_blur(input, sigma, padding_mode='replicate'):
    channels = input.shape[1]
    ksize = math.ceil(((sigma - 0.8) / 0.3 + 1) * 2 + 1)
    if ksize < 1 : ksize = 1
    ksize = math.ceil((ksize - 1) / 2) * 2 + 1
    if sigma < 1e-3: sigma = 1e-3
    kernel = torch.zeros(channels, channels, ksize, 1)
    gaussian_kernel = generate_gaussian_kernel(ksize, sigma)
    for i in range(channels):
        kernel[i, i] = gaussian_kernel
    input = torch.nn.functional.pad(input, [0, 0, (ksize - 1) // 2, (ksize - 1) // 2], padding_mode)
    input = torch.nn.functional.conv2d(input, kernel.to(input.device), stride=1)
    input = torch.nn.functional.pad(input, [(ksize - 1) // 2, (ksize - 1) // 2, 0, 0], padding_mode)
    res = torch.nn.functional.conv2d(input, kernel.reshape((channels, channels, 1, ksize)).to(input.device), stride=1)
    return res


def laplacian(input):
    channels = input.shape[1]
    kernel = torch.zeros(channels, channels, 3, 3)
    for i in range(channels):
        kernel[i, i] = torch.tensor([
            [0. , -1.,  0.],
            [-1.,  4., -1.],
            [0. , -1.,  0.],
        ])
    res = torch.nn.functional.conv2d(input, kernel.to(input.device))
    return res


bilinear_kernel_init(16)
bicubic_kernel_init(16)
