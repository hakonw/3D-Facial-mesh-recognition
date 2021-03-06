import meshfr.datasets.datasetBU3DFEv2 as datasetBU3DFEv2
import meshfr.datasets.datasetBosphorus as datasetBosphorus
import meshfr.datasets.datasetFRGC as datasetFRGC
import os

pickled = True
force = False
sample = "bruteforce" # bruteforce, 2pass, all, random
sample_size = [1024*4, 1024*6][0]


bu3dfe_path = "/lhome/haakowar/Downloads/BU_3DFE"
bu3dfe_dict =  datasetBU3DFEv2.get_bu3dfe_dict(bu3dfe_path, pickled=pickled, force=force, picke_name="/tmp/Bu3dfe-4096.p", sample="bruteforce", sample_size=1024*4)

bosphorus_path = "/lhome/haakowar/Downloads/Bosphorus/BosphorusDB"
bosphorus_dict = datasetBosphorus.get_bosphorus_dict(bosphorus_path, pickled=pickled, force=force, picke_name="/tmp/Bosphorus-4096-filter-new.p", sample=sample, sample_size=sample_size)

frgc_path = "/lhome/haakowar/Downloads/FRGCv2/Data/"
dataset_frgc_fall_2003 = datasetFRGC.get_frgc_dict(frgc_path + "Fall2003range", pickled=pickled, force=force, picke_name="/tmp/FRGCv2-fall2003_cache-4096-new.p", sample=sample, sample_size=sample_size)
dataset_frgc_spring_2003 = datasetFRGC.get_frgc_dict(frgc_path + "Spring2003range", pickled=pickled, force=force, picke_name="/tmp/FRGCv2-spring2003_cache-4096-new.p", sample=sample, sample_size=sample_size)
dataset_frgc_spring_2004 = datasetFRGC.get_frgc_dict(frgc_path + "Spring2004range", pickled=pickled, force=force, picke_name="/tmp/FRGCv2-spring2004_cache-4096-new.p", sample=sample, sample_size=sample_size)

from meshfr.datasets.dataset3DFace import get_3dface_dict
d3face_path = "/lhome/haakowar/Downloads/3DFace_DB/3DFace_DB/"
d3face_dict = get_3dface_dict(d3face_path, pickled=pickled, force=force, picke_name="/tmp/3dface-4k.p", sample="bruteforce", sample_size=4096)


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

from meshfr.io.read_wrl import read_wrl  # Data object
from meshfr.io.read_bnt import read_bnt_raw
from meshfr.io.sampler import data_all_sample
from meshfr.io.read_abs import read_abs_raw_gzip

print("Loading raw examples")
bu3dfe_example_raw = read_wrl(os.path.join(bu3dfe_path, bu3dfe_id, bu3dfe_full + "_F3D.wrl"))
bosp_example_raw = data_all_sample(read_bnt_raw(os.path.join(bosphorus_path, bosp_id, bosp_full + ".bnt")))
frgc_example_raw = data_all_sample(read_abs_raw_gzip(os.path.join(frgc_path, "Fall2003range", frgc_full + ".abs.gz")))

print("Writing raw examples")
write_off(bu3dfe_example_raw, f"./export/bu3dfe-{bu3dfe_full}-raw.off")
write_off(bosp_example_raw, f"./export/bosp-{bosp_full}-raw.off")
write_off(frgc_example_raw, f"./export/frgc-{frgc_full}-raw.off")

write_off(c(bu3dfe_example_raw), f"./export/bu3dfe-{bu3dfe_full}-raw-center.off")
write_off(c(bosp_example_raw), f"./export/bosp-{bosp_full}-raw-center.off")
write_off(c(frgc_example_raw), f"./export/frgc-{frgc_full}-raw-center.off")

print("Writing transformed model")
import meshfr.datasets.reduction_transform as reduction_transform
POST_TRANSFORM = T.Compose([T.FaceToEdge(remove_faces=False), T.Center()])
# POST_TRANSFORM_Extra = T.Compose([])
POST_TRANSFORM_Extra = T.Compose([
    reduction_transform.RandomTranslateScaled(0.01),
    T.RandomRotate(5, axis=0),
    T.RandomRotate(5, axis=1),
    T.RandomRotate(5, axis=2)
    ])

write_off(POST_TRANSFORM_Extra(POST_TRANSFORM(bu3dfe_example_raw)), f"./export/bu3dfe-{bu3dfe_full}-new-random-scale.off")
