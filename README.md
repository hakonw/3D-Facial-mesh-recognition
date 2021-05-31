# 3D Face recognition based on Geometric Deep Learning

/home/shomec/h/haakowar/miniconda3/envs/pytorch-geometric/bin/python

This is the repository for the master thesis.

This project is a proof of concept method for a new type of 3D facial recognition.

## Setup instructions

```
conda create -n pytorch-geometric python=3.8 --yes
conda activate pytorch-geometric
conda install -c pytorch pytorch=1.7.0 torchvision cudatoolkit=11.0 --yes
conda install scipy tensorboard --yes
conda install -c conda-forge trimesh pyyaml --yes
conda install -c open3d-admin open3d --yes
pip3 install addict open3d --yes # what, again?
pip3 install openmesh  # ATM is not 3.8 supported via conda, manually build (needs cmake)

pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install git+https://github.com/rusty1s/pytorch_geometric.git
pip install -e git+https://github.com/rusty1s/pytorch_geometric.git@master#egg=torch_geometric
```

This repo also uses the dataset FaceGen, which is not publicly available at the moment.

## Running

The project is tested with Cuda 11.0 and pytorch-geometric hash:3e8baf2

```
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash Anaconda3-2020.11-Linux-x86_64.sh

cd ntnuhome/git/3d-face-gdl
conda activate pytorch-geometric
python model/train.py
```

To run it, do `python model/train.py`

Run tensorboard with `tensorboard --logdir=log`
