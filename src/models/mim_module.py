from random import sample
from typing import Any, List
from einops import rearrange
import pandas as pd

import torch
from torch import nn

import torch
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau, CyclicLR

import torch.nn as nn
import segmentation_models_pytorch as smp

from src.models.components.models import BaselineUNet
from monai.networks.nets import UNETR


class MimSuperModule(LightningModule):
    """
    Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
       
        **kwargs
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()


        if self.hparams['model'] == '2dunet':
            self.model = smp.Unet(encoder_name='resnet34', in_channels=2, classes=1)
        elif self.hparams['model'] == '3dunet':
            self.model = BaselineUNet(in_channels=2, n_cls=1, n_filters=24)
        elif self.hparams['model'] == 'unetr':
            self.model = UNETR(in_channels=2, 
                               out_channels=1, 
                               img_size=(144,144,144))
        else:
            print('Please select the correct model architecture name.')
            
        self.loss_mask = Dice_and_FocalLoss()


    def forward(self, x: torch.Tensor):
        return self.model(x)

    def init_params(self, m: torch.nn.Module):
        """Initialize the parameters of a module.
        Parameters
        ----------
        m
            The module to initialize.
        Notes
        -----
        Convolutional layer weights are initialized from a normal distribution
        as described in [1]_ in `fan_in` mode. The final layer bias is
        initialized so that the expected predicted probability accounts for
        the class imbalance at initialization.
        References
        ----------
        .. [1] K. He et al. ‘Delving Deep into Rectifiers: Surpassing
           Human-Level Performance on ImageNet Classification’,
           arXiv:1502.01852 [cs], Feb. 2015.
        """

        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, a=.1)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            # initialize the final bias so that the predictied probability at
            # init is equal to the proportion of positive samples
            nn.init.constant_(m.bias, -1.5214691)


    def step(self, batch: Any):
        sample = batch
        
        if self.hparams['model'] == '2dunet':
            sample['input'] = rearrange(sample['input'], 'b c h w (sh sw) -> b c (sh h) (sw w)', sh=12, sw=12)
            sample['target_mask'] = rearrange(sample['target_mask'], 'b c h w (sh sw) -> b c (sh h) (sw w)', sh=12, sw=12)

        pred_mask = self.forward(sample['input'])
        
        loss = self.loss_mask(pred_mask, sample['target_mask'])

        return loss, pred_mask, sample['target_mask']

    def training_step(self, batch: Any, batch_idx: int):
        loss, pred_mask, target_mask = self.step(batch)

        # log train metrics
        #acc = self.train_accuracy(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        #self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, pred_mask, target_mask = self.step(batch)

        return {"loss": loss, "pred_mask": pred_mask, "target_mask": target_mask}

    def validation_epoch_end(self, outputs: List[Any]):


        loss        = torch.stack([x["loss"] for x in outputs]).mean()
        target_mask   = torch.cat([x["target_mask"] for x in outputs])
        pred_mask   = torch.cat([x["pred_mask"] for x in outputs])

        dice_val = dice(pred_mask, target_mask)

        log = {"val/loss": loss,
               "val/dice": dice_val,
               }
        
        self.log_dict(log)



        return {"loss": loss, "dice": dice_val}



    def test_step(self, batch: Any, batch_idx: int):
        

        # return self.validation_step(batch, batch_idx)
        pass

    def test_epoch_end(self, outputs: List[Any]):


        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        # optimizer = make_optimizer(AdamW, self.model, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        
        optimizer = AdamW(self.parameters(),
                         lr=self.hparams.lr,
                         weight_decay=self.hparams.weight_decay)
        scheduler = {
            "scheduler": CosineAnnealingWarmRestarts(optimizer, T_0=25, eta_min=1e-5),
            # "scheduler": MultiStepLR(optimizer, milestones=[self.hparams.step], gamma=0.1),
            #"scheduler": CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-2),
            #"scheduler": ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True),
            "monitor": "val/loss",
        }
        return [optimizer], [scheduler]
    
    

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1

    def forward(self, input, target):
        axes = tuple(range(1, input.dim()))
        intersect = (input * target).sum(dim=axes)
        union = torch.pow(input, 2).sum(dim=axes) + torch.pow(target, 2).sum(dim=axes)
        loss = 1 - (2 * intersect + self.smooth) / (union + self.smooth)
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = 1e-3

    def forward(self, input, target):
        input = input.clamp(self.eps, 1 - self.eps)
        loss = - (target * torch.pow((1 - input), self.gamma) * torch.log(input) +
                  (1 - target) * torch.pow(input, self.gamma) * torch.log(1 - input))
        return loss.mean()


class Dice_and_FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(Dice_and_FocalLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.dice_loss(input, target) + self.focal_loss(input, target)
        return loss
    

def dice(input, target):
    axes = tuple(range(1, input.dim()))
    bin_input = (input > 0.5).float()

    intersect = (bin_input * target).sum(dim=axes)
    union = bin_input.sum(dim=axes) + target.sum(dim=axes)
    score = 2 * intersect / (union + 1e-3)

    return score.mean()