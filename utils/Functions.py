'''
* Adapted from https://github.com/cwmok/LapIRN/blob/master/Code/Functions.py
'''

import itertools
import nibabel as nib
import numpy as np
import torch
import torch.utils.data as Data
import torch.nn.functional as F


# (grid[0, :, :, :, 0] - (size_tensor[3] / 2)) / size_tensor[3] * 2
def generate_grid_unit(imgshape):
    x = (np.arange(imgshape[0]) - ((imgshape[0] - 1) / 2)) / (imgshape[0] - 1) * 2
    y = (np.arange(imgshape[1]) - ((imgshape[1] - 1) / 2)) / (imgshape[1] - 1) * 2
    z = (np.arange(imgshape[2]) - ((imgshape[2] - 1) / 2)) / (imgshape[2] - 1) * 2
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid


def transform_unit_flow_to_flow(flow):
    x, y, z, _ = flow.shape
    flow[:, :, :, 0] = flow[:, :, :, 0] * (z - 1) / 2
    flow[:, :, :, 1] = flow[:, :, :, 1] * (y - 1) / 2
    flow[:, :, :, 2] = flow[:, :, :, 2] * (x - 1) / 2

    return flow


def transform_unit_flow_to_flow_2D(flow):
    b, x, y, _ = flow.shape
    flow[:, :, 0] = flow[:, :, 0] * (y - 1) / 2
    flow[:, :, 1] = flow[:, :, 1] * (x - 1) / 2

    return flow


def transform_unit_flow_to_flow_cuda(flow):
    b, x, y, z, c = flow.shape
    flow[:, :, :, :, 0] = flow[:, :, :, :, 0] * (z - 1) / 2
    flow[:, :, :, :, 1] = flow[:, :, :, :, 1] * (y - 1) / 2
    flow[:, :, :, :, 2] = flow[:, :, :, :, 2] * (x - 1) / 2

    return flow


def load_4D(name):
    # X = sitk.GetArrayFromImage(sitk.ReadImage(name, sitk.sitkFloat32 ))
    # X = np.reshape(X, (1,)+ X.shape)
    X = nib.load(name)
    X = X.get_fdata()
    X = np.reshape(X, (1,) + X.shape)
    return X


def load_4D_with_header(name):
    # X = sitk.GetArrayFromImage(sitk.ReadImage(name, sitk.sitkFloat32 ))
    # X = np.reshape(X, (1,)+ X.shape)
    X = nib.load(name)
    header, affine = X.header, X.affine
    X = X.get_fdata()
    X = np.reshape(X, (1,) + X.shape)
    return X, header, affine


def load_5D(name):
    # X = sitk.GetArrayFromImage(sitk.ReadImage(name, sitk.sitkFloat32 ))
    X = fixed_nii = nib.load(name)
    X = X.get_fdata()
    X = np.reshape(X, (1,) + (1,) + X.shape)
    return X


def imgnorm(img):
    max_v = np.max(img)
    min_v = np.min(img)
    norm_img = (img - min_v) / (max_v - min_v)
    return norm_img


def imgnorm_torch(img):
    max_v = img.max(dim=2)[0].max(dim=2)[0].unsqueeze(2).unsqueeze(2)
    min_v = img.min(dim=2)[0].min(dim=2)[0].unsqueeze(2).unsqueeze(2)
    norm_img = (img - min_v) / (max_v - min_v)
    return norm_img


def save_img(I_img, savename, header=None, affine=None):
    if header is None or affine is None:
        affine = np.diag([1, 1, 1, 1])
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    else:
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=header)

    nib.save(new_img, savename)


def save_img_nii(I_img, savename):
    # I2 = sitk.GetImageFromArray(I_img,isVector=False)
    # sitk.WriteImage(I2,savename)
    affine = np.diag([1, 1, 1, 1])
    new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    # save_path = os.path.join(output_path, savename)
    nib.save(new_img, savename)


def save_flow(I_img, savename, header=None, affine=None):
    # I2 = sitk.GetImageFromArray(I_img,isVector=True)
    # sitk.WriteImage(I2,savename)
    if header is None or affine is None:
        affine = np.diag([1, 1, 1, 1])
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    else:
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=header)

    nib.save(new_img, savename)


class Dataset(Data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, names, iterations, norm=False):
        'Initialization'
        self.names = names
        self.norm = norm
        self.iterations = iterations

    def __len__(self):
        'Denotes the total number of samples'
        return self.iterations

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        index_pair = np.random.permutation(len(self.names))[0:2]
        img_A = load_4D(self.names[index_pair[0]])
        img_B = load_4D(self.names[index_pair[1]])
        if self.norm:
            return torch.from_numpy(imgnorm(img_A)).float(), torch.from_numpy(imgnorm(img_B)).float()
        else:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float()


class Dataset_epoch(Data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, names, norm=True):
        'Initialization'
        super(Dataset_epoch, self).__init__()

        self.names = names
        self.norm = norm

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.names)

    def __getitem__(self, step):
        """Generates one sample of data"""
        # Select sample
        img = load_4D(self.names[step])

        if self.norm:
            return torch.from_numpy(imgnorm(img)).float()
        else:
            return torch.from_numpy(img).float()


class Dataset_epoch_cam(Data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, names, norm=True):
        'Initialization'
        super(Dataset_epoch_cam, self).__init__()

        self.norm = norm
        self.index_pair = list(itertools.permutations(names, 2))

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.index_pair)

    def __getitem__(self, step):
        """Generates one sample of data"""
        # Select sample
        img_A = load_4D(self.index_pair[step][0])
        img_B = load_4D(self.index_pair[step][1])

        if self.norm:
            return torch.from_numpy(imgnorm(img_A)).float(), torch.from_numpy(imgnorm(img_B)).float()
        else:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float()


class Dataset_epoch_mul(Data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, names_1, names_2, norm=True):
        """Initialization"""
        super(Dataset_epoch_mul, self).__init__()

        self.names_1 = names_1
        self.names_2 = names_2
        self.norm = norm
        self.index_pair = list(itertools.product(names_1, names_2))

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.index_pair)

    def __getitem__(self, step):
        """Generates one sample of data"""
        # Select sample
        img_A = load_4D(self.index_pair[step][0])
        img_B = load_4D(self.index_pair[step][1])

        # print(self.index_pair[step][0])
        # print(self.index_pair[step][1])

        if self.norm:
            return torch.from_numpy(imgnorm(img_A)).float(), torch.from_numpy(imgnorm(img_B)).float()
        else:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float()


class Dataset_Mapping(Data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, index_pair, norm=True):
        """Initialization"""
        super(Dataset_Mapping, self).__init__()

        self.norm = norm
        self.index_pair = index_pair

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.index_pair)

    def __getitem__(self, step):
        """Generates one sample of data"""
        # Select sample
        img_A = load_4D(self.index_pair[step][0])
        img_B = load_4D(self.index_pair[step][1])

        # print(self.index_pair[step][0])
        # print(self.index_pair[step][1])

        if self.norm:
            return torch.from_numpy(imgnorm(img_A)).float(), torch.from_numpy(imgnorm(img_B)).float()
        else:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float()


class Dataset_Mapping_val(Data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, image_pair, label_pair, norm=True):
        """Initialization"""
        super(Dataset_Mapping_val, self).__init__()

        self.norm = norm
        self.image_pair = image_pair
        self.label_pair = label_pair

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.image_pair)

    def __getitem__(self, step):
        """Generates one sample of data"""
        # Select sample
        img_A = load_4D(self.image_pair[step][0])
        img_B = load_4D(self.image_pair[step][1])

        label_A = load_4D(self.label_pair[step][0])
        label_B = load_4D(self.label_pair[step][1])

        # print(self.index_pair[step][0])
        # print(self.index_pair[step][1])

        if self.norm:
            return torch.from_numpy(imgnorm(img_A)).float(), torch.from_numpy(imgnorm(img_B)).float(), torch.from_numpy(label_A).float(), torch.from_numpy(label_B).float()
        else:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float(), torch.from_numpy(label_A).float(), torch.from_numpy(label_B).float()


class Dataset_epoch_validation(Data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, name_1, name_2, label, norm=False):
        'Initialization'
        super(Dataset_epoch_validation, self).__init__()

        self.norm = norm
        self.imgs_pair = list(itertools.product(name_1, name_2))
        self.labels_pair = list(itertools.product(label, label))

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.imgs_pair)

    def __getitem__(self, step):
        """Generates one sample of data"""
        # Select sample
        img_A = load_4D(self.imgs_pair[step][0])
        img_B = load_4D(self.imgs_pair[step][1])

        label_A = load_4D(self.labels_pair[step][0])
        label_B = load_4D(self.labels_pair[step][1])

        if self.norm:
            return torch.from_numpy(imgnorm(img_A)).float(), torch.from_numpy(imgnorm(img_B)).float(), torch.from_numpy(
                label_A).float(), torch.from_numpy(label_B).float()
        else:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float(), torch.from_numpy(
                label_A).float(), torch.from_numpy(label_B).float()


def mse_loss(input, target):
    y_true_f = input.view(-1)
    y_pred_f = target.view(-1)
    diff = y_true_f - y_pred_f
    mse = torch.mul(diff, diff).mean()
    return mse

def one_hot(seg_map, num_classes):
    seg_map = seg_map.squeeze(1).long()  # Shape becomes (1, 160, 192)

    # Apply one-hot encoding using F.one_hot and transpose to match the desired output shape
    one_hot_map = F.one_hot(seg_map, num_classes)  # Shape becomes (1, 160, 192, 4)

    # Permute the dimensions to match the desired shape (1, 4, 160, 192)
    one_hot_map = one_hot_map.permute(0, 3, 1, 2).float()
    return one_hot_map


def dice(im1, atlas):
    unique_class = np.unique(atlas)
    dice = 0
    num_count = 0
    for i in unique_class:
        if (i == 0) or ((im1 == i).sum() == 0) or ((atlas == i).sum() == 0):
            continue

        sub_dice = np.sum(atlas[im1 == i] == i) * 2.0 / (np.sum(im1 == i) + np.sum(atlas == i))
        dice += sub_dice
        num_count += 1
        # print(sub_dice)
    # print(num_count, len(unique_class)-1)
    return dice / num_count


def dice_cardiac(im1, atlas):
    unique_class = np.unique(atlas)
    myo_dice = 0
    lv_dice = 0
    rv_dice = 0
    for i in unique_class:
        if (i == 0) or ((im1 == i).sum() == 0) or ((atlas == i).sum() == 0):
            continue

        sub_dice = np.sum(atlas[im1 == i] == i) * 2.0 / (np.sum(im1 == i) + np.sum(atlas == i))
        if i == 1:
            lv_dice = sub_dice
        elif i == 2:
            myo_dice = sub_dice
        elif i == 3:
            rv_dice = sub_dice

    return myo_dice, lv_dice, rv_dice


def pair_fea(fea):
    batch = fea.shape[0]
    feas = torch.chunk(fea, batch, dim=0)
    comb_fea = list(itertools.combinations(feas, 2))

    return comb_fea

def fft_filter(X):
    _, _, H, W = X.size()
    X_fft = torch.fft.fftn(X, dim=(-2, -1))
    x_shiftn = torch.fft.fftshift(X_fft)
    X_fft_mask = x_shiftn * mask_gen_rect(H, W, hf_weights=0.0, lf_weights=1.5, radius=30)
    x_ishift = torch.fft.ifftshift(X_fft_mask)
    i_X_fft = torch.abs(torch.fft.ifftn(x_ishift, dim=(-2, -1)))
    X_fft_norm = imgnorm_torch(i_X_fft)

    return X_fft_norm


def mask_gen_rect(h, w, radius=30, hf_weights=0.0, lf_weights=1.5):
    mask = torch.full((1, 1, h, w), hf_weights).cuda()

    # Define the center coordinates
    center_x, center_y = h // 2, w // 2

    # Set the central 8x8 area to 1.5
    mask[:, :, center_x - radius:center_x + radius, center_y - radius:center_y + radius] = lf_weights

    return mask