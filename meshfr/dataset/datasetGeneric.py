from torch_geometric.data import Dataset
import random
import numpy as np

import torch_geometric.data

# Generic dataloader for face-dict of type [ident][face]

class GenericDataset(Dataset):
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
            # d.uniqid =  int("".join([str(ord(c)-38) for c in str(name)])) # Between 10 (for 0) and 84(z). Gives long error
            d.id = idx  # TODO make sure not to mix datasets
            d.name = name
            d.dataset_id = self.dataset_keys[idx]
            out.append(d)
        return torch_geometric.data.Batch.from_data_list(out)

import torch.utils.data.dataloader
import utils
# Currently, pytorch_geometric overwrites the collate_fn
# https://github.com/rusty1s/pytorch_geometric/blob/3e8baf28c86eebbf6da74be36ea3904ec77480b8/torch_geometric/data/dataloader.py#L57
# So that's a thing
class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        super(DataLoader, self).__init__(dataset, batch_size, shuffle, collate_fn=utils.list_collate_fn, **kwargs)

class ExtraTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        d = self.subset[index]
        if self.transform:
            d = self.transform(d)
        return d
        
    def __len__(self):
        return len(self.subset)