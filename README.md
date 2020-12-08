# 3D Face recognition based on Geometric Deep Learning

This is the repository for the preliminary master thesis.

This project is a proof of concept method for a new type of 3D facial recognition.

## Setup instructions

```
conda create -n pytorch3d_new python=3.8
conda activate pytorch3d_new
conda install -c pytorch pytorch=1.7.0 torchvision cudatoolkit=11.0
conda install -c conda-forge -c fvcore fvcore
conda install -c pytorch3d-nightly pytorch3d
```

This repo also uses the dataset FaceGen, which is not publicly available at the moment.

The root-dir in facegenDataset must point to the dataset folder.

## Running

The project is tested with Cuda 11.0

To run it, do `export CUBLAS_WORKSPACE_CONFIG=:4096:8; python main.py`

The export statement is to set reproducibility in the nvidia cublas api, and will not work without. Ref https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
