from torch_geometric.data import Dataset
import random
import numpy as np

# Generic dataloader for face-dict of type [ident][face]

class GenericDataset(Dataset):  # This does not need to be of type Dataset
    def __init__(self, dataset_dict: dict, transform, name_filter=lambda l: True):
        self.dataset_dict = dataset_dict
        self.dataset_keys = list(dataset_dict.keys())
        self.transform = transform
        self.filter = name_filter
        self.reduction = False

    def __len__(self):
        return len(self.dataset_keys)

    def __getitem__(self, idx):
        identity_dict = self.dataset_dict[self.dataset_keys[idx]]

        safe_dict = {}
        if self.reduction:
            identitys = random.sample(list(identity_dict.items()), min(20, len(identity_dict)))
        else:
            identitys = identity_dict.items()
        for name, data in identitys:
            if self.filter(name):
                if isinstance(data, np.ndarray):
                    safe_dict[name] = self.transform(data.copy())
                else:
                    # Clone it, so it does not get changed in the org dict
                    safe_dict[name] = self.transform(data.clone())

        # Transform into 1d array for collection  (TODO better way?)
        out = []
        for name, d in safe_dict.items():
            d.id = idx  # TODO make sure not to mix datasets
            d.name = name
            d.dataset_id = self.dataset_keys[idx]
            out.append(d)
        return out
