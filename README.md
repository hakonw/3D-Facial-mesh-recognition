# 3D Face recognition based on Geometric Deep Learning

This is the repository for the master thesis.

This project is a proof of concept method for a new type of 3D facial recognition.

## Setup instructions

```
conda create -n pytorch-geometric python=3.8
conda activate pytorch-geometric
conda install -c pytorch pytorch=1.7.0 torchvision cudatoolkit=11.0
conda install scipy

pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install torch-geometric
```

This repo also uses the dataset FaceGen, which is not publicly available at the moment.

## Running

The project is tested with Cuda 11.0

To run it, do `python main.py`
