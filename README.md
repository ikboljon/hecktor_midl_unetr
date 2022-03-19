<div align="center">

# MIDL Conference. Automatic Segmentation of Head and Neck Tumor: How Powerful Transformers Are?

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://arxiv.org/abs/2201.06251)
[![Conference](https://img.shields.io/badge/Conference-MIDL-informational)](https://www.midl.io/)
[![Dataset](https://img.shields.io/badge/Dataset-HECKTOR-blue)](https://www.aicrowd.com/challenges/miccai-2021-hecktor)
 
_Ikboljon Sobirov, Otabek Nazarov, Hussain Alasmawi, Mohammad Yaqub_
 
 The paper and reviews can be found on [OpenReview](https://openreview.net/forum?id=reIO5WfgbLd).

</div>

<br><br>

## 📌&nbsp;&nbsp;Introduction

This work was done to analyze transformers in the task of head and neck tumor segmentation. It was accepted to MIDL 2022.

## Description
**Abstract**

Cancer is one of the leading causes of death worldwide, and head and neck (H&N) cancer is amongst the most prevalent types. Positron emission tomography and computed tomography are used to detect and segment the tumor region. Clinically, tumor segmentation is extensively time-consuming and prone to error. Machine learning, and deep learning in particular, can assist to automate this process, yielding results as accurate as the results of a clinician. In this research study, we develop a vision transformers-based method to automatically delineate H\&N tumor, and compare its results to leading convolutional neural network (CNN)-based models. We use multi-modal data of CT and PET scans to do this task. We show that the selected transformer-based model can achieve results slightly below CNN-based ones. With cross validation, the model achieves a mean dice similarity coefficient of 0.736, mean precision of 0.766 and mean recall of 0.766. This is only 0.021 less than the 2020 competition winning model in terms of the DSC score. On the testing set, the model performs similarly, with DSC of 0.736, precision of 0.773, and recall of 0.760, which is only 0.023 lower in DSC than the 2020 competition winning model. This indicates that the exploration of transformer-based models is a promising research area. 

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/ikboljon/hecktor_midl_unetr
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.8
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```bash
# train on CPU
python train.py trainer.gpus=0

# train on GPU
python train.py trainer.gpus=1
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python train.py trainer.max_epochs=20 datamodule.batch_size=64
```
 
