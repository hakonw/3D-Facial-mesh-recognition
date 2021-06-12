import datasetBU3DFEv2
import datasetBosphorus
from datasetFRGC import get_frgc_dict
import os

pickled = True
force = False
sample = "2pass"
sample_size = [1024*2, 1024*6]


bu3dfe_path = "/lhome/haakowar/Downloads/BU_3DFE"
bu3dfe_dict =  datasetBU3DFEv2.get_bu3dfe_dict(bu3dfe_path, pickled=pickled, force=force, picke_name="/tmp/Bu3dfe-2048.p", sample="bruteforce", sample_size=1024*2)

bosphorus_path = "/lhome/haakowar/Downloads/Bosphorus/BosphorusDB"
bosphorus_dict = datasetBosphorus.get_bosphorus_dict(bosphorus_path, pickled=pickled, force=force, picke_name="/tmp/Bosphorus-2048-filter-new.p", sample=sample, sample_size=sample_size)

frgc_path = "/lhome/haakowar/Downloads/FRGCv2/Data/"
dataset_frgc_fall_2003 = get_frgc_dict(frgc_path + "Fall2003range", pickled=pickled, force=force, picke_name="/tmp/FRGCv2-fall2003_cache-2048-new.p", sample=sample, sample_size=sample_size)
dataset_frgc_spring_2003 = get_frgc_dict(frgc_path + "Spring2003range", pickled=pickled, force=force, picke_name="/tmp/FRGCv2-spring2003_cache-2048-new.p", sample=sample, sample_size=sample_size)
dataset_frgc_spring_2004 = get_frgc_dict(frgc_path + "Spring2004range", pickled=pickled, force=force, picke_name="/tmp/FRGCv2-spring2004_cache-2048-new.p", sample=sample, sample_size=sample_size)

# BU-3DFE: F0001_NE00WH
# Bosp: bs000_N_N_0
# frgcv2: 02463d546   # Fall2003range/02463d546.abs.gz

print("Datasets loaded")

bu3dfe_id, bu3dfe_full = "F0001", "F0001_NE00WH"
bosp_id, bosp_full = "bs000", "bs000_N_N_0"
frgc_id, frgc_full = "02463", "02463d546"

print("Loading examples")
bu3dfe_example = bu3dfe_dict[bu3dfe_id][bu3dfe_full]
bosp_example = bosphorus_dict[bosp_id][bosp_full]
frgc_example = dataset_frgc_fall_2003[frgc_id][frgc_full]

# pytorch only allowes off atm. which is acceptable. 
# It is also possible to export to trimesh and then basically every format 
# Think they are currently rewriting it to use a library instead (like meshlabs/trimesh)

from torch_geometric.io import write_off

print("Writing processed examples")
write_off(bu3dfe_example, f"./export/bu3dfe-{bu3dfe_full}.off")
write_off(bosp_example, f"./export/bosp-{bosp_full}.off")
write_off(frgc_example, f"./export/frgc-{frgc_full}.off")

import torch_geometric.transforms as T

c = T.Center()
write_off(c(bu3dfe_example), f"./export/bu3dfe-{bu3dfe_full}-center.off")
write_off(c(bosp_example), f"./export/bosp-{bosp_full}-center.off")
write_off(c(frgc_example), f"./export/frgc-{frgc_full}-center.off")

from read_wrl import read_wrl  # Data object
from read_bnt import read_bnt_raw, data_all_sample
from read_abs import read_abs_raw_gzip

print("Loading raw exanokes")
bu3dfe_example_raw = read_wrl(os.path.join(bu3dfe_path, bu3dfe_id, bu3dfe_full + "_F3D.wrl"))
bosp_example_raw = data_all_sample(read_bnt_raw(os.path.join(bosphorus_path, bosp_id, bosp_full + ".bnt")))
frgc_example_raw = data_all_sample(read_abs_raw_gzip(os.path.join(frgc_path, "Fall2003range", frgc_full + ".abs.gz")))

print("Reading raw examples")
write_off(bu3dfe_example_raw, f"./export/bu3dfe-{bu3dfe_full}-raw.off")
write_off(bosp_example_raw, f"./export/bosp-{bosp_full}-raw.off")
write_off(frgc_example_raw, f"./export/frgc-{frgc_full}-raw.off")

write_off(c(bu3dfe_example_raw), f"./export/bu3dfe-{bu3dfe_full}-raw-center.off")
write_off(c(bosp_example_raw), f"./export/bosp-{bosp_full}-raw-center.off")
write_off(c(frgc_example_raw), f"./export/frgc-{frgc_full}-raw-center.off")


