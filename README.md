# Awesome Task Sampling for Meta-Learning. 
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) 
![Static Badge](https://img.shields.io/badge/Meta_Learning-Task_Sampling-blue)
![Static Badge](https://img.shields.io/badge/to_be_continue-orange)
![Stars](https://img.shields.io/github/stars/WangJingyao07/Adaptive-Sampler)

**Official code for "Learning to Sample Tasks for Meta Learning"**

🥇🌈This repository contains not only our adaptive sampler, but also PyTorch implementation of previous samplers (Provide in the **Citation**). 

## Create Environment

For easier use and to avoid any conflicts with existing Python setup, it is recommended to use [`virtualenv`](https://docs.python-guide.org/dev/virtualenvs/) to work in a virtual environment. Now, let's start:

**Step 1:** Install [`virtualenv`](https://docs.python-guide.org/dev/virtualenvs/)

```bash
pip install --upgrade virtualenv
```

**Step 2:** Create a virtual environment, activate it:

```bash
virtualenv venv
source venv/bin/activate
```

**Step 3:** Install the requirements in [`requirements.txt`](requirements.txt).

```bash
pip install -r requirements.txt
```

## Data availability

All data sets used in this work are open source. The download and deployment ways are as follows:
​
* miniImageNet, Omniglot, and tieredImageNet will be downloaded automatically upon runnning the scripts, with the help of [pytorch-meta](https://github.com/tristandeleu/pytorch-meta).

* For ['meta-dataset'](https://github.com/google-research/meta-dataset/blob/e95c50658e4260b2ede08ede1129827b08477f1a/prepare_all_datasets.sh), follow the following steps: Download ILSVRC2012 (by creating an account [here](https://image-net.org/challenges/LSVRC/2012/index.php) and downloading `ILSVRC2012.tar`) and Cu_birds2012 (downloading from `http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz`) separately. Then, Run `sbatch scripts/download_meta_dataset/install_meta_dataset_parallel.sh` to download and prune all datasets in a parallel fashion. All the ten datasets should be copied in a single directory.

* For the few-shot-regression setting, Sinusoid, Sinusoid & Line, and Harmonic dataset are toy examples and require no downloads. Just follow the implementation in the paper.

Now, you have completed all the settings, just directly train and test as you want!!!


## Train








