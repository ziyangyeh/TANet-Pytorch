#!/usr/bin/env bash
# command to install this enviroment: source init.sh

# install miniconda3 if not installed yet.
#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
#bash Miniconda3-latest-Linux-x86_64.sh
#source ~/.bashrc

export TORCH_CUDA_ARCH_LIST="6.1;6.2;7.0;7.5;8.0;8.6"   # 3090:8.6; a100: 8.0; v100: 7.0; 2080ti: 7.5; titan xp: 6.1

pip install openmim
mim install mmcv-full
mim install mmdet
mim install mmsegmentation

pip install "git+https://github.com/open-mmlab/mmdetection3d.git"  

pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# download openpoints
git submodule update --init --recursive

# install cpp extensions, the pointnet++ library
cd openpoints/cpp/pointnet2_batch
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

pip install -r requirements.txt
