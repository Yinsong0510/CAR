import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np
import torchvision

from Functions import imgnorm_torch
from modelio import generate_grid


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        grid = generate_grid(size)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x


def calculate_jacobian_metrics(disp):
    """
    Calculate Jacobian related regularity metrics.

    Args:
        disp: (numpy.ndarray, shape (N, ndim, *sizes) Displacement field

    Returns:
        folding_ratio: (scalar) Folding ratio (ratio of Jacobian determinant < 0 points)
        mag_grad_jac_det: (scalar) Mean magnitude of the spatial gradient of Jacobian determinant
    """
    disp_n = np.moveaxis(disp, 0, -1)  # (*sizes, ndim)
    jac_det_n = calculate_jacobian_det(disp_n)
    folding_ratio = (jac_det_n < 0).sum() / np.prod(jac_det_n.shape)
    mag_grad_jac_det = np.abs(np.gradient(jac_det_n)).mean()
    return folding_ratio, mag_grad_jac_det


def calculate_jacobian_det(disp):
    """
    Calculate Jacobian determinant of displacement field of one image/volume (2D/3D)

    Args:
        disp: (numpy.ndarray, shape (*sizes, ndim)) Displacement field

    Returns:
        jac_det: (numpy.adarray, shape (*sizes) Point-wise Jacobian determinant
    """
    disp_img = sitk.GetImageFromArray(disp, isVector=True)
    jac_det_img = sitk.DisplacementFieldJacobianDeterminant(disp_img)
    jac_det = sitk.GetArrayFromImage(jac_det_img)
    return jac_det


def generate_synth(seg):
    _, _, x, y = seg.shape
    image = torch.zeros_like(seg, dtype=torch.float32)
    unique_labels = torch.unique(seg)
    for label in unique_labels:

        mask = (seg == label).float()
        num_voxels = torch.sum(mask)

        if num_voxels > 0:
            mean = np.random.uniform(25, 225)
            std = np.random.uniform(5, 25)

            normal_samples = torch.normal(mean=mean, std=std, size=seg.size()).float().cuda()

            image += mask * normal_samples
    std_gau = list(np.random.uniform(0, 1, 2))
    std_gau.sort()
    gaussian_ker = torchvision.transforms.GaussianBlur(kernel_size=5, sigma=std_gau).cuda()
    image = gaussian_ker(image)
    std_bias = np.random.uniform(0, 0.3)
    bias_field = torch.normal(mean=0, std=std_bias, size=(1, 1, 5, 6)).float().cuda()
    bias_field = torch.nn.functional.interpolate(bias_field, size=(160, 192), mode='bilinear', align_corners=True)
    bias_field = torch.exp(bias_field)
    image = image * bias_field
    image = imgnorm_torch(image)
    exp_norm = torch.normal(mean=0, std=0.25, size=(1,)).float().cuda()
    image = torch.pow(image, torch.exp(exp_norm))
    image = imgnorm_torch(image)

    return image
