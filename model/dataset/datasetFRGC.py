import os.path
from glob import glob
import pickle
from tqdm import tqdm
import sampler
from io.read_abs import read_abs_raw_gzip

def global_relevant(name): return True


def generate_frgc_dict(root, filtered=True, filter=global_relevant, sample="bruteforce", sample_size=2048):
    dataset = {}
    
    # There are no sub folders, as for every date, every ident is in one folder
    # Find all sub-folders
    # Which are all the identities

    file_path_list = sorted(glob(os.path.join(root, "*.abs.gz")))
    assert(len(file_path_list) > 0)
    # 04717d43.abs.gz

    def add(identity, basename, value):
        if identity not in dataset:
            dataset[identity] = {}
        dataset[identity][basename] = value

    for file_path in tqdm(file_path_list):
        basename = os.path.basename(file_path)
        basename = basename[:-7]  # Hardcoded, remove  .abs.gz
        identity = basename.split("d")[0]

        # Skip if the file is not relevant & the filter is on
        if filtered and not filter(basename):
            continue
            
        data_raw = read_abs_raw_gzip(file_path)
        if sample == "2pass": data_sampled = sampler.data_2pass_sample(data_raw, sample_size[0], sample_size[1])
        elif sample == "bruteforce": data_sampled = sampler.data_bruteforce_sample(data_raw)
        elif sample == "random": data_sampled = sampler.data_simple_sample(data_raw, sample_size)
        elif sample == "all": data_sampled = sampler.data_all_sample(data_raw)
        else: raise ValueError("Invalid argument", sample)
        # identity_data[basename] = data_raw

        add(identity, basename, data_sampled)


    return dataset


# DOES NOT CHECK IF FILTER IS CHANGED
# TODO maybe save entire dataset, followed by applying filter post?
# Or possibly both
def get_frgc_dict(root, pickled, force=False, picke_name="FRGCv2_cache.p", filter=global_relevant, sample="2pass", sample_size=2048):
    if pickled and not force:
        try:
            print("Loading pickle")
            dataset = pickle.load(open(picke_name, "rb"))
            print("Pickle loaded")
            return dataset
        except Exception as e:
            print(f"Pickle failed - {str(e)}, loading data manually")

    dataset = generate_frgc_dict(root, filtered=True, filter=filter, sample=sample, sample_size=sample_size)

    if pickled:
        print("Saving pickle")
        pickle.dump(dataset, open(picke_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)
        print("Pickle saved")

    return dataset



if __name__ == "__main__":
    dataset = get_frgc_dict(
        "/lhome/haakowar/Downloads/FRGCv2/Data/Fall2003range",
        pickled=True,
        force=False,
        picke_name="FRGCv2-fall2003_cache.p"
        )
    print(len(dataset))
    print(list(dataset.keys()))

    dataset = get_frgc_dict(
        "/lhome/haakowar/Downloads/FRGCv2/Data/Spring2003range",
        pickled=True,
        force=False,
        picke_name="FRGCv2-spring2003_cache.p"
        )
    print(len(dataset))
    print(list(dataset.keys()))

    dataset = get_frgc_dict(
        "/lhome/haakowar/Downloads/FRGCv2/Data/Spring2004range",
        pickled=True,
        force=False,
        picke_name="FRGCv2-spring2004_cache.p"
    )
    print(len(dataset))
    print(list(dataset.keys()))

    if False:
        data = dataset["bs104"]["bs104_N_N_3"]
        # data = dataset_cached["M0008"]["M0008_AN01WH"]
        print(data)
        import torch_geometric.utils
        trimesh = torch_geometric.utils.to_trimesh(data)
        trimesh.export("bs104_N_N_3-2k.ply")
