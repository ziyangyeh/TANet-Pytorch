#!/usr/bin/env bash
# command to install this enviroment: source init.sh

# install miniconda3 if not installed yet.
#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
#bash Miniconda3-latest-Linux-x86_64.sh
#source ~/.bashrc

export TORCH_CUDA_ARCH_LIST="6.1;6.2;7.0;7.5;8.0;8.6"   # 3090:8.6; a100: 8.0; v100: 7.0; 2080ti: 7.5; titan xp: 6.1
module purge
module load cuda/11.3.1
module load gcc
# make sure local cuda version is 11.1

# download openpoints
git submodule update --init --recursive

# install PyTorch
conda deactivate
conda env remove --name yzy
conda create -n yzy -y python=3.8 numpy numba
conda activate yzy

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
# install relevant packages
# torch-scatter is a must, and others are optional
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
# pip install torch-scatter torch-sparse torch-cluster torch-geometric -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
pip install -r PointNeXt/requirements.txt

# install cpp extensions, the pointnet++ library
cd PointNeXt/openpoints/cpp/pointnet2_batch
python setup.py install
cd ../

# grid_subsampling library. necessary only if interested in S3DIS_sphere
cd subsampling
python setup.py build_ext --inplace
cd ..


# # point transformer library. Necessary only if interested in Point Transformer and Stratified Transformer
cd pointops/
python setup.py install
cd ..


# Blow are functions that optional. Necessary only if interested in reconstruction tasks such as completion
cd chamfer_dist
python setup.py install --user
cd ../emd
python setup.py install --user
cd ../../../
