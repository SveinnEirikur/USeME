import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn.functional import conv2d, pad

import pytorch_lightning as pl

from einops import asnumpy, rearrange, reduce, repeat
import numpy as np

from mat_loaders import get_data
from models import MTMx, create_conv_kernel, normalize_channels
from utilities import gaussian_filters

from resize_right import resize


class S2ImageDataset(Dataset):
    def __init__(self, filename, datadir, bicubic=False, transform=None, target_transform=None, d = [6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2]):
        self.train = False
        yim, mtf, xm_im = get_data(filename, datadir=datadir, get_mtf=True)[:3]

        self.d = d

        if bicubic:
            yim = torch.stack([resize(torch.tensor(y).unsqueeze(0), scale_factors=s).squeeze() for y, s in zip(yim, self.d)])
        else:
            yim = torch.stack([repeat(torch.tensor(y), 'h w -> (h sh) (w sw)', sh=s, sw=s) for y, s in zip(yim, self.d)])

        self.target = rearrange(torch.tensor(xm_im), 'h w c -> c h w')
        self.target_shape = self.target.shape
        self.yim, self.ch_mean = normalize_channels(yim)

        sdf = np.array(d)*np.sqrt(-2*np.log(mtf)/np.pi**2)
        self.sigmas = np.array(sdf)

        fft_of_B = create_conv_kernel(self.sigmas, self.yim[1].shape[-2], self.yim[1].shape[-1], d=self.d, N=[18, 0, 0, 0, 12, 12, 12, 0, 12, 18, 12, 12])
        self.fft_of_B = fft_of_B

        self.datadir = datadir
        self.filename = filename
        self.transform = transform
        self.target_transform = target_transform

        self.num_samples = 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx >= self.num_samples:
            return None

        image = self.yim
        target = self.target

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return target, image, self.fft_of_B, self.ch_mean


def downsample_s2_ims(or_ims, psfs, s, d, fft_of_B, n_pad=3):

    out_shape = or_ims[(d == s).nonzero()[0]].shape

    tr = torch.stack([repeat(torch.tensor(y), 'h w -> (h sh) (w sw)', sh=sd, sw=sd) for y, sd in zip(or_ims, d)])
    tr = reduce(tr, 'c (h sh) (w sw) -> c h w', 'mean', sh=s, sw=s)

    assert(np.allclose(asnumpy(tr[(d == s).nonzero()[0]]), or_ims[(d == s).nonzero()[0]]))

    for ds in d.unique().sort()[0][d.unique() > s]:
        lr = torch.stack([torch.as_tensor(or_ims[idx]) for idx in (d == ds).nonzero()])
        mr = torch.stack([tr[idx].squeeze() for idx in (d < ds).nonzero()])


        padding = (n_pad, n_pad, n_pad, n_pad)
        mr_ims = rearrange(mr, 'c h w -> 1 c h w')
        mr_ims_pad = pad(mr_ims, padding, mode="circular")

        im_ds_flat = rearrange(lr, 'c h w -> 1 (h w) c')

        psf = rearrange(psfs[(d < ds).nonzero()].squeeze(), 'c h w -> c 1 h w').type_as(mr_ims_pad)

        mr_ds = conv2d(mr_ims_pad,
                       psf,
                       stride=int(torch.div(ds, s, rounding_mode='floor')),
                       groups=psf.shape[0])

        mr_ds_flat = rearrange(mr_ds, '1 c h w -> 1 (h w) c')

        mr_ds_oned = pad(mr_ds_flat, (1, 0, 0, 0, 0, 0), 'constant', 1)

        lls = torch.linalg.lstsq(mr_ds_oned, im_ds_flat).solution.squeeze().T

        lr_at_mr = repeat(rearrange(lls @ pad(rearrange(mr,
                                                'c h w -> c (h w)'),
                                    (0, 0, 1, 0), value=1),
                            'c (h w) -> c h w', h=mr.shape[-2], w=mr.shape[-1]),
                        'c h w -> c (h sh) (w sw)', sh=s, sw=2)

        fft_B = create_conv_kernel([0.2,0.2], s*out_shape[-2], s*out_shape[-1], d=[s,s], N=[12,12])

        lr_at_mr = torch.real(torch.fft.ifft2(fft_B*torch.fft.fft2(lr_at_mr)))
        lr_at_mr = reduce(MTMx(lr_at_mr, [s]*lr_at_mr.shape[0]).squeeze(), 'c (h sh) (w sw) -> c h w', 'max', sh=s, sw=s)

        tr[(d == ds).nonzero().squeeze()] = lr_at_mr

    tr_ds = torch.real(torch.fft.ifft2(fft_of_B*torch.fft.fft2(tr)))
    tr_ds = torch.stack([repeat(reduce(MTMx(tr_MB, [ds]).squeeze(), '(h sh) (w sw) -> h w', 'max', sh=ds, sw=ds), 'h w -> (h sh) (w sw)', sh=ds, sw=ds) for ds, tr_MB in zip(d, tr_ds)])
    target = tr
    image = tr_ds

    return image, target


class S2ImageTrainingDataset(Dataset):
    def __init__(self, filename, datadir, s=1, bicubic=False, n_pad=3, augment=True, patch_repeat=4, transform=None, target_transform=None, d = [6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2]):
        self.train = False
        yim, mtf, xm_im = get_data(filename, datadir=datadir, get_mtf=True)[:3]

        self.d = torch.as_tensor(d)

        self.target = rearrange(torch.tensor(xm_im), 'h w c -> c h w')

        self.patch_repeat = patch_repeat

        while yim[0].shape[-1]%6:
            if yim[0].shape[1]%2:
                yim[0] = yim[0][:, :-1]
                yim[9] = yim[9][:, :-1]
                yim[1] = yim[1][:, :-6]
                yim[2] = yim[2][:, :-6]
                yim[3] = yim[3][:, :-6]
                yim[7] = yim[7][:, :-6]
                yim[4] = yim[4][:, :-3]
                yim[5] = yim[5][:, :-3]
                yim[6] = yim[6][:, :-3]
                yim[8] = yim[8][:, :-3]
                yim[10] = yim[10][:, :-3]
                yim[11] = yim[11][:, :-3]
                self.target = self.target[:, :, :-6]
            else:
                yim[0] = yim[0][:, 1:]
                yim[9] = yim[9][:, 1:]
                yim[1] = yim[1][:, 6:]
                yim[2] = yim[2][:, 6:]
                yim[3] = yim[3][:, 6:]
                yim[7] = yim[7][:, 6:]
                yim[4] = yim[4][:, 3:]
                yim[5] = yim[5][:, 3:]
                yim[6] = yim[6][:, 3:]
                yim[8] = yim[8][:, 3:]
                yim[10] = yim[10][:, 3:]
                yim[11] = yim[11][:, 3:]
                self.target = self.target[:, :, 6:]

        while yim[0].shape[-2]%6:
            if yim[0].shape[1]%2:
                yim[0] = yim[0][:-1, :]
                yim[9] = yim[9][:-1, :]
                yim[1] = yim[1][:-6, :]
                yim[2] = yim[2][:-6, :]
                yim[3] = yim[3][:-6, :]
                yim[7] = yim[7][:-6, :]
                yim[4] = yim[4][:-3, :]
                yim[5] = yim[5][:-3, :]
                yim[6] = yim[6][:-3, :]
                yim[8] = yim[8][:-3, :]
                yim[10] = yim[10][:-3, :]
                yim[11] = yim[11][:-3, :]
                self.target = self.target[:, :-6, :]
            else:
                yim[0] = yim[0][1:, :]
                yim[9] = yim[9][1:, :]
                yim[1] = yim[1][6:, :]
                yim[2] = yim[2][6:, :]
                yim[3] = yim[3][6:, :]
                yim[7] = yim[7][6:, :]
                yim[4] = yim[4][3:, :]
                yim[5] = yim[5][3:, :]
                yim[6] = yim[6][3:, :]
                yim[8] = yim[8][3:, :]
                yim[10] = yim[10][3:, :]
                yim[11] = yim[11][3:, :]
                self.target = self.target[:, 6:, :]

        self.yim = yim

        sdf = np.array(d)*np.sqrt(-2*np.log(mtf)/np.pi**2)
        self.sigmas = np.array(sdf)

        self.psf = torch.tensor(gaussian_filters(n_pad*2+s, self.sigmas))

        fft_of_B = create_conv_kernel(self.sigmas, self.target.shape[-2]//s, self.target.shape[-1]//s, d=self.d, N=[18, 0, 0, 0, 12, 12, 12, 0, 12, 18, 12, 12])
        if s > 1:
            image, target = downsample_s2_ims(yim, self.psf, s, d=self.d, fft_of_B=fft_of_B, n_pad=n_pad)
        else:
            image = torch.stack([repeat(torch.tensor(y), 'h w -> (h sh) (w sw)', sh=sd, sw=sd) for y, sd in zip(self.yim, d)])
            target = self.target

        if augment:
            target = repeat(target, 'c h w -> c (2 h) (2 w)')
            target = rearrange(target, 'c (h1 h) (w1 w) -> (h1 w1) c h w', h1=2, w1=2)
            target[1] = target[1].flip(-2)
            target[2] = target[2].flip(-1)
            target[3] = target[3].flip((-2,-1))

            image = repeat(image, 'c h w -> c (2 h) (2 w)')
            image = rearrange(image, 'c (h1 h) (w1 w) -> (h1 w1) c h w', h1=2, w1=2)
            image[1] = image[1].flip(-2)
            image[2] = image[2].flip(-1)
            image[3] = image[3].flip((-2,-1))
        else:
            image = rearrange(image, 'c h w -> 1 c h w')
            target = rearrange(target, 'c h w -> 1 c h w')

        hn = np.min([24,image.shape[-2]])
        while image.shape[-2]%hn != 0:
            hn=hn+1
        hs = image.shape[-2]//hn

        wn = np.min([24,image.shape[-1]])
        while image.shape[-1]%wn != 0:
            wn=wn+1
        ws = image.shape[-1]//wn

        self.y = rearrange(image, 'b c (h1 h) (w1 w) -> (h1 w1 b) c h w', h1=hs, w1=ws).float()
        self.x = rearrange(target, 'b c (h1 h) (w1 w) -> (h1 w1 b) c h w', h1=hs, w1=ws).float()

        if augment:
            cn = np.min([hn, wn])
            self.y = self.y[:,:,:cn,:cn]
            self.y = torch.cat((rearrange(self.y, 'b c h w -> b c w h'), self.y))
            self.x = self.x[:,:,:cn,:cn]
            self.x = torch.cat((rearrange(self.x, 'b c h w -> b c w h'), self.x))

        self.fft_of_B = create_conv_kernel(self.sigmas, self.y.shape[-2], self.y.shape[-1], d=self.d, N=[18, 0, 0, 0, 12, 12, 12, 0, 12, 18, 12, 12])

        assert(self.x.shape == self.y.shape)

        self.y, self.ch_means = normalize_channels(self.y)

        self.datadir = datadir
        self.filename = filename
        self.transform = transform
        self.target_transform = target_transform

        self.num_samples = self.y.shape[0]

    def __len__(self):
        return self.num_samples * self.patch_repeat

    def __getitem__(self, idx):
        if idx >= self.num_samples * self.patch_repeat:
            return None

        image = self.y[idx%self.num_samples]
        target = self.x[idx%self.num_samples]
        ch_mean = self.ch_means[idx%self.num_samples]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return target, image, self.fft_of_B, ch_mean


class S2ImageModule(pl.LightningDataModule):
    def __init__(self, datadir: str = "../dataset", trainfile: str = "apex", valfile: str = "apex", testfile: str = "apex", batch_size: int = 8, patch_repeat=4, num_workers: int = 24, bicubic: bool = False):
        super().__init__()
        self.datadir = datadir
        self.trainfile = trainfile
        self.testfile = testfile
        self.valfile = valfile
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.bicubic = bicubic
        self.patch_repeat = patch_repeat

    def setup(self, stage: str=None):
        self.s2_test = S2ImageDataset(self.testfile, self.datadir, bicubic=self.bicubic)
        self.s2_train_2 = S2ImageTrainingDataset(self.trainfile, self.datadir, 2, bicubic=self.bicubic, patch_repeat=self.patch_repeat)
        self.s2_val = S2ImageDataset(self.valfile, self.datadir, bicubic=self.bicubic)

    def train_dataloader(self):
        return DataLoader(self.s2_train_2, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, prefetch_factor=2*self.batch_size) #, \

    def val_dataloader(self):
        return DataLoader(self.s2_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.s2_test, batch_size=1, num_workers=self.num_workers)



