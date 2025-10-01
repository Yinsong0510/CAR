import functools
import inspect
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn

from contrast_aug import GINGroupConv


def store_config_args(func):
    """
    Class-method decorator that saves every argument provided to the
    function as a dictionary in 'self.config'. This is used to assist
    model loading - see LoadableModel.
    """

    attrs, varargs, varkw, defaults = inspect.getargspec(func)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self.config = {}

        # first save the default values
        if defaults:
            for attr, val in zip(reversed(attrs), reversed(defaults)):
                self.config[attr] = val

        # next handle positional args
        for attr, val in zip(attrs[1:], args):
            self.config[attr] = val

        # lastly handle keyword args
        if kwargs:
            for attr, val in kwargs.items():
                self.config[attr] = val

        return func(self, *args, **kwargs)

    return wrapper


class LoadableModel(nn.Module):
    """
    Base class for easy pytorch model loading without having to manually
    specify the architecture configuration at load time.

    We can cache the arguments used to the construct the initial network, so that
    we can construct the exact same network when loading from file. The arguments
    provided to __init__ are automatically saved into the object (in self.config)
    if the __init__ method is decorated with the @store_config_args utility.
    """

    # this constructor just functions as a check to make sure that every
    # LoadableModel subclass has provided an internal config parameter
    # either manually or via store_config_args
    def __init__(self, *args, **kwargs):
        if not hasattr(self, 'config'):
            raise RuntimeError('models that inherit from LoadableModel must decorate the '
                               'constructor with @store_config_args')
        super().__init__(*args, **kwargs)

    def save(self, path):
        """
        Saves the model configuration and weights to a pytorch file.
        """
        # don't save the transformer_grid buffers - see SpatialTransformer doc for more info
        sd = self.state_dict().copy()
        grid_buffers = [key for key in sd.keys() if key.endswith('.grid')]
        for key in grid_buffers:
            sd.pop(key)
        torch.save({'config': self.config, 'model_state': sd}, path)

    @classmethod
    def load(cls, path, device):
        """
        Load a python model configuration and weights.
        """
        checkpoint = torch.load(path, map_location=torch.device(device))
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state'], strict=False)
        return model


def default_unet_features():
    nb_features = [
        [16, 32, 32, 32],  # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
    return nb_features


def merge(feats):
    merged = torch.cat((feats[0], feats[1]), dim=1)
    for level, feat in enumerate(feats):
        if level > 1:
            merged = torch.nn.functional.max_pool2d(merged, kernel_size=2, stride=2)
            merged = torch.cat((merged, feat), dim=1)
    return merged


def calculate_support(inshape, threshold):
    grid = generate_grid((20, 24))

    x_cor = grid[0, 0, :, :].reshape(-1, 1).unsqueeze(0).unsqueeze(0).cuda()
    y_cor = grid[0, 1, :, :].reshape(-1, 1).unsqueeze(0).unsqueeze(0).cuda()

    img_cor = torch.cat((x_cor[0], y_cor[0]), dim=2)

    distance_matrix = torch.sum(img_cor ** 2, -1, keepdim=True)
    distance_matrix = distance_matrix + torch.sum(img_cor ** 2, -1, keepdim=True).transpose(1, 2)
    distance_matrix = distance_matrix - 2 * torch.bmm(img_cor, img_cor.transpose(1, 2))

    support = (distance_matrix < threshold ** 2)[0].float().cuda()

    return support


def generate_grid(inshape):
    vectors = [torch.arange(0, s) for s in inshape]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids)
    grid = torch.unsqueeze(grid, 0)
    grid = grid.type(torch.FloatTensor)

    return grid


class FFDGenerator(nn.Module):
    def __init__(self, size, ffd_spacing, **kwargs):
        super(FFDGenerator, self).__init__()
        self.size = size
        self.dimension = len(self.size)
        if isinstance(ffd_spacing, (tuple, list)):
            if len(ffd_spacing) == 1:
                ffd_spacing = ffd_spacing * self.dimension
            assert len(ffd_spacing) == self.dimension
            self.ffd_spacing = ffd_spacing
        elif isinstance(ffd_spacing, (int, float)):
            self.ffd_spacing = (ffd_spacing,) * self.dimension
        else:
            raise NotImplementedError
        self.kwargs = kwargs
        img_spacing = kwargs.pop('img_spacing', None)
        if img_spacing is None:
            self.img_spacing = [1] * self.dimension
        else:
            if isinstance(img_spacing, (tuple, list, np.ndarray)):
                assert len(img_spacing) == self.dimension
                self.img_spacing = img_spacing
            elif isinstance(img_spacing, (int, float)):
                self.img_spacing = (img_spacing,) * self.dimension
            else:
                raise NotImplementedError

        self.control_point_size = [int((self.size[i] * self.img_spacing[i] - 1) // self.ffd_spacing[i] + 4)
                                   for i in range(self.dimension)]

        vectors = [torch.arange(0., self.size[i] * self.img_spacing[i], self.img_spacing[i]) / self.ffd_spacing[i]
                   for i in range(self.dimension)]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid_floor = torch.floor(grid)
        grid_decimal = grid - grid_floor
        self.register_buffer('grid_floor', grid_floor, persistent=False)
        self.register_buffer('grid_decimal', grid_decimal, persistent=False)

        mesh_indices = torch.stack(torch.meshgrid(*[torch.arange(4)] * self.dimension)).flatten(1)
        self.register_buffer('mesh_indices', mesh_indices.T, persistent=False)

    def forward(self, mesh):
        mesh_shape = mesh.shape[2:]
        assert len(mesh_shape) == self.dimension
        assert all([mesh_shape[i] == self.control_point_size[i] for i in range(self.dimension)]), \
            "Expected control point size %s, got %s!" % (self.control_point_size, list(mesh_shape))

        flow = torch.zeros(*mesh.shape[:2], *self.size, dtype=mesh.dtype, device=mesh.device)
        for idx in self.mesh_indices:
            B = self.Bspline(self.grid_decimal, idx)
            pivots = self.grid_floor.squeeze(0) + idx.view(self.dimension, *[1] * self.dimension)
            pivots = pivots.to(torch.int64)
            if self.dimension == 2:
                flow += B.prod(dim=1, keepdim=True) * mesh[:, :, pivots[0], pivots[1]]
            elif self.dimension == 3:
                flow += B.prod(dim=1, keepdim=True) * mesh[:, :, pivots[0], pivots[1], pivots[2]]
            else:
                raise NotImplementedError
        return flow

    def Bspline(self, decimal, idx):
        idx = idx.view(self.dimension, *[1] * self.dimension).unsqueeze(0)

        return torch.where(idx == 0,
                           (1 - decimal) ** 3 / 6,
                           torch.where(idx == 1,
                                       decimal ** 3 / 2 - decimal ** 2 + 2 / 3,
                                       torch.where(idx == 2,
                                                   - decimal ** 3 / 2 + decimal ** 2 / 2 + decimal / 2 + 1 / 6,
                                                   torch.where(idx == 3,
                                                               decimal ** 3 / 6,
                                                               torch.zeros_like(decimal)
                                                               )
                                                   )
                                       )
                           )


class Sobel_Filter(nn.Module):
    def __init__(self):
        super(Sobel_Filter, self).__init__()

        # Sobel kernels for edge detection
        self.sobel_x = torch.tensor([[1, 0, -1],
                                     [2, 0, -2],
                                     [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.sobel_y = torch.tensor([[1, 2, 1],
                                     [0, 0, 0],
                                     [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Register the kernels as buffers so they are moved to the appropriate device during training
        self.register_buffer('sobel_x_filter', self.sobel_x)
        self.register_buffer('sobel_y_filter', self.sobel_y)

    def forward(self, x):
        # Apply Sobel filters for edge detection in both directions
        grad_x = F.conv2d(x, self.sobel_x_filter, padding=1)
        grad_y = F.conv2d(x, self.sobel_y_filter, padding=1)

        # Combine the gradients
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        return grad_magnitude


class aug_image(torch.nn.Module):
    def __init__(self):
        super(aug_image, self).__init__()

        self.RandConv = GINGroupConv()
        self.sobel = Sobel_Filter()
        self.threshold = torch.Tensor([0.24]).cuda()

    def forward(self, x):
        while True:
            x = self.RandConv(x)
            edge = torch.mean(self.sobel(x))
            if edge > self.threshold:
                break

        return x
