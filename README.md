# 3D Face recognition based on Geometric Deep Learning

This is the repository for the master thesis.

This project is a proof of concept method for a new type of 3D facial recognition.

## Setup instructions

Check the avaiable cuda versions with `ls /usr/local/ | grep cuda` and match the pytorch and pytorch-geometric packages to this. This is important for it to run properly.  

If you want to install it in an existing venv, match the pytorch-geometric with versions with
```
python -c "import torch; print(torch.version.cuda)"
python -c "import torch; print(torch.__version__)"
```
<<<<<<< HEAD
conda create -n pytorch3d_new python=3.8
conda activate pytorch3d_new
conda install -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
conda install -c conda-forge -c fvcore -c iopath fvcore iopath
conda install -c pytorch3d-nightly pytorch3d
=======

>>>>>>> develop
```
conda create -n pytorch-geometric python=3.8 --yes
conda activate pytorch-geometric
# Standard pytorch install. Follow pytorch install instructions for latest
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia --yes
# Extra packages needed by this repo and pytorch-geometric
conda install scipy tensorboard --yes
conda install -c conda-forge trimesh pyyaml matplotlib --yes

# These packages are required for pre-processing the dataset and subsampling.
# If you use the "all" sampler or "random" sampler, they are not needed
# ATM is 3.8 NOT supported via conda, manually build open3d (needs cmake installed)
conda install -c open3d-admin open3d --yes
conda install pandas --yes
pip3 install addict open3d
pip3 install openmesh

# Install pytorch_geometric
# If there are any problems. Use the latest compatible pytorch, cuda and lates pytorch-geometric.
#   Install information and latest versions can be found at  https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip3 install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip3 install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip3 install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip3 install torch-geometric

```

To run the project, the codebase MUST be installed as a package. 
```
# Install codebase as a package. Must be inside the git root
pip install -e .
```

Either install the latest released pytorch_geometric via:
`pip install torch-geometric`
or install master `pip3 install -e git+https://github.com/rusty1s/pytorch_geometric.git@master#egg=torch_geometric`.  
Master may be unstable, but contain some newer fixes. During development, only master had a fix for bug for the pooling.

## Running

The project is tested with Cuda 11.0, pytorch 1.7.0, and pytorch-geometric hash:3e8baf2  
and Cuda 11.1, pytorch 1.8.1 and pytorch-geometric hash:32519a5  (v1.7.0)
Cuda 11.2 has issues has pytorch is skipping that cuda version. 

```
# CD into the roof folder and activate enviroment
cd ntnuhome/git/3d-face-gdl
conda activate pytorch-geometric

# To run it with logging, do
python meshfr/testing.py NAME-OF-EXPERIMENT
```
It can also be run without logging with `python meshfr/testing.py q`



Run tensorboard with `tensorboard --logdir=logdir --port 6006`. Replace logdir with the newest logging directory

