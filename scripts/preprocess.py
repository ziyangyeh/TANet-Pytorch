import sys, os
sys.path.append(os.getcwd())

import argparse
from glob import glob
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from tqdm_batch import batch_process
from tqdm import tqdm
from tqdm.contrib import tzip

import vedo
import numpy as np

from utils import rearrange, extract_teeth_without_gingiva, to_origin_and_normalize, global_rotation_matrix

CSV_SAVE_PATH = "dataset_csv"

def load_and_process(dir_path, out_path=None, num: int=50):
    upper_mesh_path = glob(dir_path+"上/*_arch_upper_aligned.stl")[0]
    lower_mesh_path = glob(dir_path+"下/*_arch_lower_aligned.stl")[0]
    upper_label_path = glob(dir_path+"上/*.txt")[0]
    lower_label_path = glob(dir_path+"下/*.txt")[0]
    upper_mesh = vedo.Mesh(upper_mesh_path)
    lower_mesh = vedo.Mesh(lower_mesh_path)
    upper_mesh.celldata["Label"]=rearrange(np.loadtxt(upper_label_path))
    lower_mesh.celldata["Label"]=rearrange(np.loadtxt(lower_label_path))
    mesh = vedo.merge(upper_mesh, lower_mesh)
    mesh = extract_teeth_without_gingiva(mesh)
    label = mesh.celldata["Label"].astype(np.int64)
    mesh = to_origin_and_normalize(mesh.to_trimesh())
    transform_matrices = global_rotation_matrix(num)
    if out_path is None:
        out_path = os.path.join(*dir_path.split("/")[:-2], "augmented")
    else:
        out_path = os.path.join(*dir_path.split("/")[:-2], out_path)
    if os.path.exists(out_path):
        pass
    else:
        os.mkdir(out_path)
    for i in range(num):
        res = vedo.utils.trimesh2vedo(mesh.apply_transform(transform_matrices[i]))
        res.celldata["Label"]=label
        vedo.io.write(res, os.path.join(out_path, dir_path.split("/")[-2]+"_%02d.vtp"%i))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-process and augmentation dataset.")
    parser.add_argument("-r", "--dir_root", type=str, metavar="", help="dataset directory")
    parser.add_argument("-o", "--out_root", type=str, metavar="", help="output directory")
    parser.add_argument("-n", "--aug_num", type=int, metavar="", help="augmentation times", default=20)

    args = parser.parse_args()
    
    DATA_ROOT = glob(args.dir_root+"/*[0-9]/")
    if args.out_root is None:
        batch_process(
            DATA_ROOT,
            load_and_process,
            n_workers=cpu_count(),
            sep_progress=True,
        )
        out_root = "augmented"
    else:
        Parallel(n_jobs=cpu_count())(delayed(load_and_process)(*item) for _, item in enumerate(tqdm(tzip(DATA_ROOT, [args.out_root]*len(DATA_ROOT), [args.aug_num]*len(DATA_ROOT)))))
        out_root = args.out_root

    import pandas as pd
    data_set = pd.DataFrame(np.asarray([[os.path.join(root, file) for file in files] for root, dirs, files in os.walk(os.path.join(args.dir_root, out_root))])[0])
    if os.path.exists(CSV_SAVE_PATH):
        pass
    else:
        os.mkdir(CSV_SAVE_PATH)
    data_set.to_csv(os.path.join(CSV_SAVE_PATH,"data_set.csv"), index=False)
