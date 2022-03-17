<div align="center">

# MIDL Conference. Automatic Segmentation of Head and Neck Tumor: How Powerful Transformers Are?

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://arxiv.org/abs/2201.06251)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://www.midl.io/)
 
_Ikboljon Sobirov, Otabek Nazarov, Hussain Alasmawi, Mohammad Yaqub_
 
 The paper can be found on [OpenReview](https://openreview.net/forum?id=reIO5WfgbLd).

</div>

<br><br>

## 📌&nbsp;&nbsp;Introduction

This work was done to analyze transformers in the task of head and neck tumor segmentation. It was accepted to MIDL 2022.

## Description

What it does

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
 
Check