import torch
from torch import nn
from torch.nn.functional import l1_loss
from torch.nn import L1Loss
from torch.optim import NAdam

from pytorch_lightning import LightningModule

from einops import rearrange, reduce, repeat, asnumpy

import wandb

import json

import os

import numpy as np

from utilities import gaussian_filter, gaussian_filters, NumpyEncoder
from custom_metrics import MeanAbsoluteGradientError
from sreval import evaluate_performance


def MTMx(x, d):
    d = torch.tensor(d)
    M = torch.zeros([1, len(d), x.shape[-2], x.shape[-1]]).type_as(x)
    for s in torch.unique(d):
        M[:, d == s,::s,::s]=1
    y = M*x
    return y


def normMx(x, d):
    d = torch.tensor(d)
    y = x.clone()
    M = torch.zeros([1, len(d), x.shape[-2], x.shape[-1]]).type_as(x)
    for s in torch.unique(d):
        M[:,d == s,::s,::s]=1
    Mx = M*x
    vn = torch.zeros(x.shape[:2]).type_as(x)*torch.nan
    for s in torch.unique(d):
        if s != 1:
            b = Mx[:,d==s,:,:]
            b = reduce(b, 'b c (h sh) (w sw) -> b c h w', 'max', sh=s, sw=s)
            vn[:,d==s] = torch.linalg.vector_norm(b, dim=(-2,-1))
    return vn


def create_conv_kernel(sdf, nl, nc, d=[6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2], N=[18, 0, 0, 0, 12, 12, 12, 0, 12, 18, 12, 12]):
    L = len(d)
    B = torch.zeros([L, nl, nc])
    for i in range(L):
        if d[i] == 1 or N[i] == 0:
            B[i,0,0] = 1
        else:
            h = torch.tensor(gaussian_filter(N[i],sdf[i]))
            B[i, int(nl%2+(nl - N[i])//2):int(nl%2+(nl + N[i])//2), int(nc%2+(nc - N[i])//2):int(nc%2+(nc + N[i])//2)] = h

            B[i,:,:] = torch.fft.fftshift(B[i,:,:])
            B[i,:,:] = torch.divide(B[i,:,:],torch.sum(B[i,:,:]))
    FBM = torch.fft.fft2(B)

    return FBM


def normalize_channels(x, ch_mean=None, unnormalize=None):
    # Mean squared power is 1
    if unnormalize is None:
        if ch_mean is None:
            ch_mean = reduce(x.square(), '... h w -> ... 1 1', 'mean')
        y = torch.sqrt(x.square()/ch_mean)
        return y, ch_mean
    else:
        y = torch.sqrt(x.square()*unnormalize)
        return y


class S2Filters(LightningModule):

    '''
    S2 Filter Unrolling
    '''

    def __init__(self, filter_size: int = 3,
                       sub_size: int = 12,
                       d: list = [6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2]):

        '''
        filter_size (int): Size of each filter in the group
        '''
        super(S2Filters, self).__init__()

        self.input_size = len(d)
        self.w_layer = torch.nn.Conv2d(len(d),sub_size,filter_size,padding='same',padding_mode='circular', bias=False)
        self.wT_layer = lambda x: torch.nn.functional.conv_transpose2d(x, self.w_layer.weight, padding = filter_size//2)


    def forward(self, x_t):
        '''
        inputs: (torch.(cuda.)Tensor) Model inputs: tensor of input image 
        B x C x H x W


        returns: (torch.(cuda.)Tensor) Output of the filter with learnable
        regularization kernels B x C x H x W
        '''

        u = self.w_layer(x_t)
        x = self.wT_layer(u)

        assert(not x.isnan().any())

        return x, u


class SimpleMultiBandRegularized(LightningModule):

    def __init__(self,
                 n_filters: int = 1,
                 filter_size: int = 3,
                 beta: list = [0.9, 0, 0, 0, 0.05, 0.05, 0.05, 0, 0.05, 0.9, 0.05, 0.05],
                 learn_beta: bool = False,
                 alpha: list = [-2, -100, -100, -100, -4, -4, -4, -100, -4, -2, -4, -4],
                 learn_alpha: bool = False,
                 mtf: list = [.32, .26, .28, .24, .38, .34, .34, .26, .33, .26, .22, .23],
                 d: list = [6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2],
                 output_size: tuple = (252, 252)):

        '''
        n_filters (int): Number of learnable filters in a group
        filter_size (int): Size of each filter in the group
        '''
        super(SimpleMultiBandRegularized, self).__init__()

        self.output_size = output_size
        if len(alpha) == 1:
            alpha = repeat(torch.tensor(alpha), 'c -> 1 (nc c) 1 1', nc=len(d))
        else:
            alpha = rearrange(torch.tensor(alpha), 'c -> 1 c 1 1')
        self.alpha = nn.Parameter(alpha, requires_grad=learn_alpha)
        if len(beta) == 1:
            beta = repeat(torch.tensor(beta), 'c -> 1 (nc c) 1 1', nc=len(d))
        else:
            beta = rearrange(torch.tensor(beta), 'c -> 1 c 1 1')
        self.beta = nn.Parameter(beta, requires_grad=learn_beta)

        self.regularization = S2Filters(filter_size=filter_size, sub_size=n_filters, d=d)

        self.d = d

        sdf = np.array(d)*np.sqrt(-2*np.log(mtf)/np.pi**2)
        self.sigmas = np.array(sdf)
        self.psfs = torch.tensor(gaussian_filters(18, self.sigmas), dtype=torch.float64)

        N = [max(12, s*3) if s > 1 else 0 for s in d]

        fft_of_B = create_conv_kernel(self.sigmas, self.output_size[-2], self.output_size[-1], d=self.d, N=N)
        self.register_buffer('fft_of_B', torch.view_as_real(fft_of_B.unsqueeze(0)))

    def forward(self, inputs):
        """
        Function that performs one iteration of the gradient descent scheme of the deconvolution algorithm.

        x: (torch.(cuda.)Tensor) Image, restored with the previous iteration of the gradient descent scheme, B x C x H x W
        y: (torch.(cuda.)Tensor) Input blurred and noisy image B x C x H x W

        (self.) alpha: (torch.(cuda.)Tensor) Power of a trade-off coefficient exp(alpha)
        (self.) beta: (torch.(cuda.)Tensor) Gradient descent step size
        returns: (torch.(cuda.)Tensor) Restored image B x C x H x W
        """

        (x, y, fft_of_B, alpha, beta) = inputs

        if fft_of_B == None:
            fft_of_B = torch.view_as_complex(self.fft_of_B)

        if alpha is not None:
            self.alpha = nn.Parameter(
                torch.FloatTensor([alpha], device=self.device))

        if beta is not None:
            self.beta = nn.Parameter(
                torch.FloatTensor([beta], device=self.device))
            
        x_t, latent = self.regularization(x)

        regul = torch.exp(self.alpha) * x_t
        xy_filtered = torch.real(torch.fft.ifft2(fft_of_B*torch.fft.fft2(x)))-y
        xy_resampled = MTMx(xy_filtered, self.d)
        xy_refiltered = torch.real(torch.fft.ifft2(fft_of_B.conj()*torch.fft.fft2(xy_resampled)))

        brackets = xy_refiltered + regul
        out = x - self.beta * brackets

        out = (out, y, fft_of_B, alpha, beta)

        return out


class SteepestDescentZS(LightningModule):
    """
    Module that uses UNet to predict individual gradient of a regularizer for each input image and then
    applies gradient descent scheme with predicted gradient of a regularizers per-image
    """
    def __init__(
            self,
            batch_size: int = 1,
            lr: float = 1e-3,
            gamma: float = 0.5,
            lamb: float = 0.5,
            rho: float = 10000.0,
            sigma: float = 1.0,
            taper: bool = False,
            alpha: float = [-4.6, -100.0, -100.0, -100.0, -4.6, -4.6, -4.6, -100.0, -4.6, -4.6, -4.6, -4.6],
            learn_alpha: bool =True,
            beta: float = None,
            learn_beta: bool = False,
            regularizer: str = 'igarss',
            n_filters: int = 1,
            filter_size: int = 3,
            mtf: list = [.32, .26, .28, .24, .38, .34, .34, .26, .33, .26, .22, .23],
            d: list = [6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2],
            beta_1: float = 0.98,
            beta_2: float = 0.9999, 
            sched_gamma: float = 1.0,
            output_size: tuple = (198, 198),
            n_its: int = 10,
            eval_metrics: bool = True,
            save_test_pred: bool = False
    ):

        super(SteepestDescentZS, self).__init__()
        
        self.function = SimpleMultiBandRegularized
        if beta is None:
            beta = [2.0, 0, 0, 0, 2.0, 2.0, 2.0, 0, 2.0, 2.0, 2.0, 2.0]

        self.n_its = n_its
        
        self.mod_layers = [self.function(alpha=alpha,
                                         learn_alpha=learn_alpha,
                                         beta=beta,
                                         learn_beta=learn_beta,
                                         n_filters=n_filters,
                                         filter_size=filter_size,
                                         mtf=mtf,
                                         d=d,
                                         output_size=output_size) for i in range(self.n_its)]

        self.seq_model = nn.Sequential(*self.mod_layers)
        
        self.reg_name = regularizer
    
        self.gradient_loss = MeanAbsoluteGradientError(gamma)
        self.l1_loss = L1Loss()
        self.lamb = lamb
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.sched_gamma = sched_gamma
        self.batch_size = batch_size
        self.learning_rate = lr
        self.d = torch.tensor(d)
        self.rho = rho
        self.sigma = sigma
        
        self.taper = taper
        self.eval_metrics = eval_metrics
        self.save_test_pred = save_test_pred

    def forward(self, y, fft_of_B=None, alpha=None, beta=None):
        """
        y: (torch.(cuda.)Tensor) Tensor of input images of shape B x C x H x W
        returns: (torch.(cuda.)Tensor) Output of the gradient descent scheme B x C x H x W
        """
        
        x = y.clone()

        inputs = (x, y, fft_of_B, alpha, beta)
        (x,
         y,
         fft_of_B,
         alpha,
         beta) = self.seq_model(inputs)

        return x

    def training_step(self, batch, batch_idx):
        target_20, image_20, fft_of_B_20, ch_mean_20 = batch
        
        pred_20 = self.forward(image_20, fft_of_B=fft_of_B_20)
        
        pred_20 = normalize_channels(pred_20, unnormalize=ch_mean_20)
        step_l1_loss_20 = l1_loss(pred_20[:,[0, 4, 5, 6, 8, 9, 10, 11],6:-6,6:-6], target_20[:,[0, 4, 5, 6, 8, 9, 10, 11],6:-6,6:-6])
        step_g_loss_20 = self.gradient_loss(pred_20[:,[0, 4, 5, 6, 8, 9, 10, 11],6:-6,6:-6], target_20[:,[0, 4, 5, 6, 8, 9, 10, 11],6:-6,6:-6])
        step_loss_20 = step_l1_loss_20 + self.lamb * step_g_loss_20

        step_loss = step_loss_20

        self.log_dict({"step_loss_20": step_loss_20,
                        "l1_loss": step_l1_loss_20, 
                        "g_loss": step_g_loss_20 
                        }, prog_bar=True)
        return step_loss


    def test_step(self, batch, batch_idx):
        target, image, fft_of_B, ch_mean = batch
        x = normalize_channels(image.clone(), unnormalize=ch_mean)
        pred = self.forward(image, fft_of_B=fft_of_B)
        pred = normalize_channels(pred, unnormalize=ch_mean)
        
        z = torch.stack([pred[:, 0, :, :],
                         x[:, 1, :, :], 
                         x[:, 2, :, :], 
                         x[:, 3, :, :], 
                         pred[:, 4, :, :], 
                         pred[:, 5, :, :], 
                         pred[:, 6, :, :], 
                         x[:, 7, :, :], 
                         pred[:, 8, :, :], 
                         pred[:, 9, :, :], 
                         pred[:, 10, :, :], 
                         pred[:, 11, :, :]])

        if self.save_test_pred:
            run_name = wandb.run.name
            np.savez(self.reg_name + '_' + run_name +'.npz', x_hat=asnumpy(z))

        if self.eval_metrics:
            [ssim, sre, rmse, ergas, sam, uiqi] = evaluate_performance(asnumpy(rearrange(target.squeeze().clamp(0,10000), 'c h w -> h w c')), asnumpy(rearrange(z.squeeze().clamp(0,10000), 'c h w -> h w c')), data_range=10000.0, limsub=6)

            self.logger.log_metrics({"SSIM": ssim,
                    "SRE": sre,
                    "RMSE": rmse,
                    "ERGAS": ergas,
                    "SAM": sam,
                    "UIQI": uiqi})

            with open(os.path.join(wandb.run.dir, "results.json"), 'w') as f:
                json.dump({"SSIM": ssim,
                        "SRE": sre,
                        "RMSE": rmse,
                        "ERGAS": ergas,
                        "SAM": sam,
                        "UIQI": uiqi}, f, cls=NumpyEncoder)

        pred = z.squeeze()
        self.logger.log_image(key="Predicted", images=[pred[0].clamp(0, 10000)/10000, pred[1].clamp(0, 10000)/10000, pred[2].clamp(0, 10000)/10000, pred[3].clamp(0, 10000)/10000, pred[4].clamp(0, 10000)/10000, pred[5].clamp(0, 10000)/10000, pred[6].clamp(0, 10000)/10000, pred[7].clamp(0, 10000)/10000, pred[8].clamp(0, 10000)/10000, pred[9].clamp(0, 10000)/10000, pred[10].clamp(0, 10000)/10000, pred[11].clamp(0, 10000)/10000],
                              caption=["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"])
        if self.eval_metrics:
            target = target.squeeze()
            self.logger.log_image(key="Target", images=[target[0].clamp(0, 10000)/10000, target[1].clamp(0, 10000)/10000, target[2].clamp(0, 10000)/10000, target[3].clamp(0, 10000)/10000, target[4].clamp(0, 10000)/10000, target[5].clamp(0, 10000)/10000, target[6].clamp(0, 10000)/10000, target[7].clamp(0, 10000)/10000, target[8].clamp(0, 10000)/10000, target[9].clamp(0, 10000)/10000, target[10].clamp(0, 10000)/10000, target[11].clamp(0, 10000)/10000],
                                caption=["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"])

    def validation_step(self, batch, batch_idx):
        target, image, fft_of_B, ch_mean = batch

        pred = self.forward(image, fft_of_B=fft_of_B)

        pred = normalize_channels(pred, unnormalize=ch_mean)

        step_l1_loss = l1_loss(pred[:,[0, 4, 5, 6, 8, 9, 10, 11],6:-6,6:-6], target[:,[0, 4, 5, 6, 8, 9, 10, 11],6:-6,6:-6])
        step_g_loss = self.gradient_loss(pred[:,[0, 4, 5, 6, 8, 9, 10, 11],6:-6,6:-6], target[:,[0, 4, 5, 6, 8, 9, 10, 11],6:-6,6:-6])
        
        self.log_dict({"val_l1_loss": step_l1_loss,
                       "val_g_loss": step_g_loss})

        loss_val = step_l1_loss + self.lamb * step_g_loss

        self.log_dict({"val_loss": loss_val},
                      prog_bar=True)

        return {"val_loss": loss_val}

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x["val_loss"] for x in outputs]).mean()
        log_dict = {"val_loss": loss_val}
        return {"log": log_dict, "val_loss": log_dict["val_loss"], "prog_bar": log_dict}

    def training_epoch_end(self, outputs):
        loss_val = torch.stack([x["loss"] for x in outputs]).mean()
        self.log_dict({"train_loss": loss_val}, prog_bar=True)

    def configure_optimizers(self):
        params = [*self.parameters()]
        opt = NAdam(params, lr=self.learning_rate, betas=(self.beta_1, self.beta_2))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[2,4,8,16,32,64,128], gamma=self.sched_gamma)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
                "strict": False
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }
