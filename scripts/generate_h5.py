import sys, os
sys.path.append(os.getcwd())

import h5py
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from data import pl_teeth_dataset, TeethDataset
from openpoints.utils import EasyConfig

def _gen_dataset(input_len, teeth_num, sample_num, num_workers, dataset: TeethDataset, data_type: str, file: h5py.File):
    X_shape = (input_len, teeth_num*sample_num, 3)
    X_v_shape = (input_len, teeth_num, sample_num, 3)
    C_shape = (input_len, teeth_num+2, 3)
    dof_shape = (input_len, teeth_num, 6)
    data_group = file.create_group(data_type)
    data_group.create_dataset("X", data=np.zeros(X_shape, dtype=np.float32), dtype="float32")
    data_group.create_dataset("X_v", data=np.zeros(X_v_shape, dtype=np.float32), dtype="float32")
    data_group.create_dataset("target_X", data=np.zeros(X_shape, dtype=np.float32), dtype="float32")
    data_group.create_dataset("target_X_v", data=np.zeros(X_v_shape, dtype=np.float32), dtype="float32")
    data_group.create_dataset("C", data=np.zeros(C_shape, dtype=np.float32), dtype="float32")
    data_group.create_dataset("6dof", data=np.zeros(dof_shape, dtype=np.float32), dtype="float32")
    for idx, item in enumerate(tqdm(DataLoader(dataset, num_workers=num_workers), desc=data_type+'_data')):
        data_group["X"][idx]=item["X"].squeeze(0).numpy()
        data_group["X_v"][idx]=item["X_v"].squeeze(0).numpy()
        data_group["target_X"][idx]=item["target_X"].squeeze(0).numpy()
        data_group["target_X_v"][idx]=item["target_X_v"].squeeze(0).numpy()
        data_group["C"][idx]=item["C"].squeeze(0).numpy()
        data_group["6dof"][idx]=item["6dof"].squeeze(0).numpy()

def gen_h5(cfg_path, out_dir):
    if os.path.exists(out_dir):
        pass
    else:
        os.mkdir(out_dir)

    if os.path.exists(os.path.join(out_dir,'TeethDataset.hdf5')):
        os.remove(os.path.join(out_dir,'TeethDataset.hdf5'))

    cfg = EasyConfig()
    cfg.load(cfg_path, recursive=True)
    cfg = cfg.LitDataModule
    sample_num = cfg.dataset.sample_num
    teeth_num = cfg.dataset.teeth_num
    num_workers = cfg.dataloader.num_workers

    train_data, val_data, test_data = pl_teeth_dataset.LitDataModule(cfg).enum_dataset()

    f = h5py.File(os.path.join(out_dir,'TeethDataset.hdf5'),'a')

    _gen_dataset(train_data.__len__(), teeth_num, sample_num, num_workers, train_data, data_type="train", file=f)

    _gen_dataset(val_data.__len__(), teeth_num, sample_num, num_workers, val_data, data_type="val", file=f)

    _gen_dataset(test_data.__len__(), teeth_num, sample_num, num_workers, test_data, data_type="test", file=f)

    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="H5 Dataset Script")
    parser.add_argument("-c", "--cfg_path", type=str, metavar="", help="configuration file path", default="config/default.yaml")
    parser.add_argument("-o", "--out_dir", type=str, metavar="", help="H5 dateset out directory", default="tmp/teeth_seg/augmented/h5")

    args = parser.parse_args()

    gen_h5(args.cfg_path, args.out_dir)
