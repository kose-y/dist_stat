# dist_stat 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/kose-y/dist_stat/master?filepath=pytorch-dist-workshop.ipynb)
 __← Click for tutorial__


This repository contains the Python package `dist_stat` for distributed computation based on `distributed` submodule of PyTorch. The package includes the distributed matrix data structure (`distmat`) and operations on it. In addition, the following applications of the data structure are also included:
- Nonnegative Matrix Factorization (NMF)
- Multidimensional Scaling (MDS)
- Positron Emission Tomography (PET)
- L1-penalized Cox Regression.

This package targets solving high-dimensional statistical problems on single-node multi-GPU systems, multi-node clusters, and supercomputers with minimal changes in the code. It is tested on single-node multi-GPU workstations with up to 8 GPUs and virtual clusters created on the Amazon Web Services (AWS) with up to 20 nodes with 36 physical CPU cores each (720 cores in total), with the data size up to ~200,000 x 500,000.

## Installation
### Prerequisites
#### Python (>= 3.6)
Our code is tested on Python 3.6 installed through [anaconda](https://www.anaconda.com/distribution/#download-section). We highly recommend using anaconda, because compilation instruction for PyTorch assumes using it.

#### CUDA >= 9.0, cuDNN >= 7.0 (for GPU support)
For multi-GPU support, the following are required:
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) v9 or above,
- [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v7 or above.

#### (CUDA-aware) MPI
MPI is a de facto standard communication directives for tightly-coupled high-performance computing. Any MPI implementations that run in PyTorch is okay, such as Open MPI or MPICH. However, if multi-GPU support is desired, the MPI should be ["CUDA-aware"](https://devblogs.nvidia.com/introduction-cuda-aware-mpi/). For CUDA-aware support on OpenMPI, it should be compiled from source, following [these](https://www.open-mpi.org/faq/?category=buildcuda) directions.
We tested our code using OpenMPI v1.10-4.0. 

#### PyTorch (compiled from source)
For PyTorch to support MPI, it should be compiled [from source](https://github.com/pytorch/pytorch#from-source). We tested our code on PyTorch versions 0.4-1.3.

### Installing the package
One can run the following commands to install the package after installing the prerequisites:
```
git clone https://github.com/kose-y/dist_stat.git
cd dist_stat
python setup.py install
```

## Running the examples
See [Examples](./examples).

## Deployment on the AWS
How to deploy the virtual cluster on the AWS using [AWS ParallelCluster](https://docs.aws.amazon.com/parallelcluster/) is discussed in Appendix C of our paper below. We used up to 20 `c5.18xlarge` instances on AWS EC2. The job scripts to submit the jobs to the Sun Grid Engine are given in [Jobs](./jobs).

## Acknowledgement
This work was supported by [AWS Cloud Credits for Research](https://aws.amazon.com/research-credits/). This research has been conducted using the UK Biobank Resource under application number 48152.

## Citation

Ko S, Zhou H, Zhou J, and Won J-H (2019+). High-Performance Statistical Computing in the Computing Environment of the 2020s. [arXiv:2001.01916](https://arxiv.org/abs/2001.01916).
