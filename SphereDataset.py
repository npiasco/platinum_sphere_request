import torch.utils as utils
import torch.utils.data
import torchvision.transforms.functional as func
import torchvision as torchvis
import pandas as pd
import PIL.Image
import numpy as np
import scipy.misc


class ToTensor(torchvis.transforms.ToTensor):
    def __init__(self):
        torchvis.transforms.ToTensor.__init__(self)

    def __call__(self, sample):
        for name, mod in sample.items():
            sample[name] = func.to_tensor(mod).float()

        return sample


class Resize(torchvis.transforms.Resize):
    def __init__(self, size, interpolation=PIL.Image.BILINEAR):
        torchvis.transforms.Resize.__init__(self, size, interpolation=interpolation)

    def __call__(self, sample):
        for name, mod in sample.items():
            if name in ['rgb']:
                sample[name] = func.resize(mod, self.size, self.interpolation)
            else:
                sample[name] = func.resize(mod, self.size, PIL.Image.NEAREST)

        return sample


class SphereDataset(utils.data.Dataset):
    def __init__(self, root, file, modalities, **kwargs):
        self.root = root
        self.transform = kwargs.pop('transform', None)
        self.bearing = kwargs.pop('bearing', True)
        self.panorama_split = kwargs.pop('panorama_split', {'v_split': 3,
                                                            'h_split': 2,
                                                            'offset': 0})

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        self.data = pd.read_csv(self.root + file, header=1, sep=',')
        self.modalities = modalities
        self.used_mod = self.modalities

    def __len__(self):
        s = self.data.__len__()
        if self.panorama_split is not None:
            s *= self.panorama_split['h_split'] * self.panorama_split['v_split']
        return s

    def __getitem__(self, idx):
        sample = dict()
        for mod_name in self.used_mod:
            if self.panorama_split is not None:
                split_idx = idx % (self.panorama_split['h_split'] * self.panorama_split['v_split'])
                fidx = idx // (self.panorama_split['h_split'] * self.panorama_split['v_split'])
            else:
                fidx = idx
            file_name = self.root + self.data.ix[fidx, self.mod_to_indx(mod_name)] + '.png'
            if self.panorama_split is not None:
                raw_img = scipy.misc.imread(file_name)
                r = raw_img.shape[1] / (2 * np.pi)
                vert_ang = np.degrees(raw_img.shape[0]/r)
                v_pas_angle = (360 / self.panorama_split['v_split'])
                h_pas_angle = (vert_ang / self.panorama_split['h_split'])
                offset = self.panorama_split['offset']

                v_im_num = split_idx % self.panorama_split['v_split']
                h_im_num = split_idx // self.panorama_split['v_split']

                size_im = [int(2 * np.tan(np.radians(h_pas_angle/2)) * r),
                           int(2 * np.tan(np.radians(v_pas_angle/2)) * r),
                           3]
                color_img = np.zeros((size_im[0], size_im[1], size_im[2]), np.uint8)
                for j in range(size_im[0]):
                    for i in range(size_im[1]):
                        if i >= size_im[1]:
                            x_angle = np.radians(offset + v_im_num * v_pas_angle + v_pas_angle/2) + \
                                      np.arctan((i - size_im[1]/2)/r)
                        else:
                            x_angle = np.radians(offset + v_im_num * v_pas_angle + v_pas_angle/2) - \
                                      np.arctan((size_im[1]/2 - i)/r)
                        if j >= size_im[0]:
                            y_angle = np.radians(h_im_num * h_pas_angle + h_pas_angle / 2) + \
                                      np.arctan((j - size_im[0]/2)/r)
                        else:
                            y_angle = np.radians(h_im_num * h_pas_angle + h_pas_angle / 2) - \
                                      np.arctan((size_im[0]/2 - j)/r)
                        im_j = int(r * y_angle)
                        im_i = int(r * x_angle)
                        color_img[j, i, :] = raw_img[im_j, im_i, :]

                sample[mod_name] = PIL.Image.fromarray(color_img)
            else:
                sample[mod_name] = PIL.Image.open(file_name)

        if self.transform:
            if 'first' in self.transform:
                sample = torchvis.transforms.Compose(self.transform['first'])(sample)
            for mod in self.transform:
                if mod not in ('first',) and mod in self.used_mod:
                    sample[mod] = torchvis.transforms.Compose(self.transform[mod])({mod: sample[mod]})[mod]

        sample['idx'] = fidx
        return sample

    @staticmethod
    def mod_to_indx(mod):
        return {'rgb': 3, 'depth': 4, 'sem': 5}.get(mod)
