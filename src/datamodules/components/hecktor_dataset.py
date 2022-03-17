import numpy as np
import nibabel as nib
import pathlib
import os
import pandas as pd
from torch.utils.data import Dataset
from einops import rearrange
import torch

def get_paths_to_patient_files(path_to_imgs, PatientID, append_mask=True):
        """
    Get paths to all data samples, i.e., CT & PET images (and a mask) for each patient.

    Parameters
    ----------
    path_to_imgs : str
        A path to a directory with patients' data. Each folder in the directory must corresponds to a single patient.
    append_mask : bool
        Used to append a path to a ground truth mask.

    Returns
    -------
    list of tuple
        A list wherein each element is a tuple with two (three) `pathlib.Path` objects for a single patient.
        The first one is the path to the CT image, the second one - to the PET image. If `append_mask` is True,
        the path to the ground truth mask is added.
    """
        path_to_imgs = pathlib.Path(path_to_imgs)

        patients = [p for p in os.listdir(path_to_imgs) if os.path.isdir(path_to_imgs / p)]
        paths = []
        for p in patients:
            path_to_ct = path_to_imgs / p / (p + '_ct.nii.gz')
            path_to_pt = path_to_imgs / p / (p + '_pt.nii.gz')

            if append_mask:
                path_to_mask = path_to_imgs / p / (p + '_ct_gtvt.nii.gz')
                paths.append((path_to_ct, path_to_pt, path_to_mask))
            else:
                paths.append((path_to_ct, path_to_pt))
        return paths



class HecktorDataset(Dataset):

    def __init__(self, paths_to_samples, transform=None, mode='train'):
        
        self.paths_to_samples = paths_to_samples
        self.transform = transform
        self.patient_id = self.get_patients_id()
        self.paths = get_paths_to_patient_files(paths_to_samples, self.patient_id)

        if mode not in ['train', 'test']:
            raise ValueError(f"Argument 'mode' must be 'train' or 'test'. Received {mode}")
        self.mode = mode

        if mode == 'train':
            self.num_of_seqs = len(self.paths[0]) - 1
        else:
            self.num_of_seqs = len(self.paths[0])


    def get_patients_id(self):

        df = pd.read_csv('/home/ikboljonsobirov/hecktor/TUSurv/data/hecktor2021_patient_info_training.csv')
        clinical_data = df
        patient_id = clinical_data['PatientID']
        
        return patient_id

    def __len__(self):
        return len(self.paths)


    def __getitem__(self, index):
        sample = dict()
        
        id_ = self.paths[index][0].parent.stem

        sample['id'] = id_
        img = [self.read_data(self.paths[index][i]) for i in range(self.num_of_seqs)]
        img = np.stack(img, axis=-1)
        # img = rearrange(img,'h w d c -> c h w d')
        sample['input'] = img #np.expand_dims(img, axis=0)
        
        mask = self.read_data(self.paths[index][-1])
        mask = np.expand_dims(mask, axis=3)
        # mask = rearrange(mask,'h w d c->c h w d')
        sample['target_mask'] = mask
        
        if self.transform:
            sample = self.transform(sample)
                
        return sample


    @staticmethod
    def read_data(path_to_nifti, return_numpy=True):
        """Read a NIfTI image. Return a numpy array (default) or `nibabel.nifti1.Nifti1Image` object"""
        if return_numpy:
            return nib.load(str(path_to_nifti)).get_fdata()
        return nib.load(str(path_to_nifti))