from typing import Optional
from sklearn.model_selection import KFold, StratifiedKFold

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, Subset
from torchvision import transforms


from src.datamodules.augmentations import *

from src.datamodules.components.hecktor_dataset import HecktorDataset
import pandas as pd

class HECKTORDataModule(LightningDataModule):
    """
    Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()

        self.data_dir = self.hparams["data_dir"]
        #self.train_val_test_split = self.hparams["train_val_test_split"]
        self.batch_size = self.hparams["batch_size"]
        self.num_workers = self.hparams["num_workers"]
        self.pin_memory = self.hparams["pin_memory"]
        self.Fold = self.hparams["Fold"]

        self.train_transforms = transforms.Compose([
                                Mirroring(p=0.5),
                                RandomRotation(p=0.5, angle_range=[0, 45]),
                                AdjustContrast(gamma=1),
                                ElasticDeformation(),
                                NormalizeIntensity(),
                                ToTensor() 
                            ])

        self.test_transforms = transforms.Compose([
                               NormalizeIntensity(), 
                               ToTensor()
                            ])




        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None


    def setup(self, stage: Optional[str] = None):
        
        self.dataset = HecktorDataset(self.hparams["data_dir"],
                                      transform=self.train_transforms, 
                                      mode='train')
        
        full_indices = range(len(self.dataset))

        kf = KFold(n_splits=5, shuffle=True, random_state=786)

        train_idx = {}
        test_idx = {}

        key = 1
        for i,j in kf.split(full_indices):
            train_idx[key] = i
            test_idx[key] = j

            key += 1

        train_dataset, val_dataset = Subset(self.dataset, train_idx[self.Fold]), Subset(self.dataset, test_idx[self.Fold])
        val_dataset.dataset.transform = self.test_transforms

        self.data_train = train_dataset
        self.data_val = val_dataset
        
        

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )
    

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=True,
        )


