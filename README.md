# 3D Face recognition based on Geometric Deep Learning

This is the repository for the master thesis.

This project is a proof of concept method for a new type of 3D facial recognition.

## Setup instructions

```
conda create -n pytorch-geometric python=3.8
conda activate pytorch-geometric
conda install -c pytorch pytorch=1.7.0 torchvision cudatoolkit=11.0
conda install scipy tensorboard

pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install git+https://github.com/rusty1s/pytorch_geometric.git
```

This repo also uses the dataset FaceGen, which is not publicly available at the moment.

## Running

The project is tested with Cuda 11.0 and pytorch-geometric hash:3e8baf2

To run it, do `python model/train.py`

Run tensorboard with `tensorboard --logdir=log`
