import os.path
from glob import glob
import pickle
from tqdm import tqdm
from meshfr.io.read_bnt import read_bnt_raw
import meshfr.io.sampler as sampler


# global_relevant = lambda name: "_N_N_" in name or "_E_" in name
# global_relevant = lambda name: "IGN" not in name
global_relevant = lambda name: "_N_N_" in name or "_LFAU_" in name or "_UFAU_" in name or "_CAU_" in name or "_E_" in name
def generate_bosphorus_dict(root, filtered=True, filter=global_relevant, sample="bruteforce", sample_size=2048):
    dataset = {}

    # Find all sub-folders
    # Which are all the identities
    folders = []
    for root, dirs, filenames in os.walk(root):
        folders = sorted(dirs)
        break  # prevent descending into subfolders

    pbar = tqdm(folders)
    for folder in pbar:
        pbar.set_description(f"Processing {folder}")
        file_path_list = sorted(glob(os.path.join(root, folder, "*.bnt")))
        assert(len(file_path_list) > 0)

        identity_data = {}
        for file_path in file_path_list:
            basename = os.path.basename(file_path)
            basename = basename[:-4]  # Hardcoded, remove  .bnt

            # Skip if the file is not relevant & the filter is on
            if filtered and not filter(basename):
                continue

            data_raw = read_bnt_raw(file_path)
            if sample == "2pass": data_sampled = sampler.data_2pass_sample(data_raw, sample_size[0], sample_size[1])
            elif sample == "bruteforce": data_sampled = sampler.data_bruteforce_sample(data_raw)
            elif sample == "random": data_sampled = sampler.data_simple_sample(data_raw, sample_size)
            elif sample == "all": data_sampled = sampler.data_all_sample(data_raw)
            else: raise ValueError("Invalid argument", sample)
            identity_data[basename] = data_sampled
            # identity_data[basename] = data_raw

        dataset[folder] = identity_data
    
    return dataset


# DOES NOT CHECK IF FILTER IS CHANGED
# TODO maybe save entire dataset, followed by applying filter post?
# Or possibly both
def get_bosphorus_dict(root, pickled, force=False, picke_name="Bosphorus_cache.p", filter=global_relevant, sample="2pass", sample_size=2048):
    if pickled and not force:
        try:
            print("Loading pickle")
            dataset = pickle.load(open(picke_name, "rb"))
            print("Pickle loaded")
            return dataset
        except Exception as e:
            print(f"Pickle failed - {str(e)}, loading data manually")
    
    dataset = generate_bosphorus_dict(root, filtered=True, filter=filter, sample=sample, sample_size=sample_size)

    if pickled:
        print("Saving pickle")
        pickle.dump(dataset, open(picke_name, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        print("Pickle saved")
    
    return dataset


# Neutral Pose & Expression: N
#    1 Pose  (N_N)
# Lower Face Action Unit: LFAU
#   20 Poses 
# Upper Face Action Unit: UFAU
#    5 Poses 
# Action Unit Combination: CAU
#    3 Poses 
# Emotional Expression: E
#    6 Poses
# Yaw Rotation: YR
#    7 Poses
# Pitch Rotation: PR
#    4 Poses
# Cross Rotation: CR
#    2 Poses
# Occlusion: O
#    4 Poses
# Ignored: IGN
#    1 Pose

# From "Deep 3D Face Identification" 
#  Bosphorus The Bosphorus database contains
#  4,666 3D facial scans over 105 subjects with
#  rich expression variations, poses, occlusions.
#  The 2,902 scans contain expression variations
#  from 105 subjects. In the experiment, 105 first
#  neutral scans from each identity are used as
#  a gallery set and 2797 non-neutral scans are
#  used as a probe set. BU-3DFE The BU-3DFE
#  database contains 2500 3D non-neutral scans
#  are used as a probe set.

if __name__ == "__main__":
    dataset = get_bosphorus_dict("/lhome/haakowar/Downloads/Bosphorus/BosphorusDB", pickled=True, force=False)
    print(len(dataset))

    if False:
        data = dataset["bs104"]["bs104_N_N_3"]
        # data = dataset_cached["M0008"]["M0008_AN01WH"]
        print(data)
        import torch_geometric.utils
        trimesh = torch_geometric.utils.to_trimesh(data)
        trimesh.export("bs104_N_N_3-2k.ply")
