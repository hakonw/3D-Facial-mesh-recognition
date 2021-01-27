from torch_geometric.data import Data
import torch_geometric.utils
import torch
from tqdm import tqdm

#
# Processing of data before metrics
#

@torch.no_grad()  # This function disables autograd, so no training can be done on the data
def single_data_to_descriptor(model, device, data):
    data.to(device)
    # Assuming single data, ergo dont need to do .to_data_list()
    descriptor = model(data)
    descriptor.to("cpu")  # Results are returned to the cpu, as ASAPooling would use over 12gb memory (because of grad)
    return descriptor

# Assumes dict: ID -> list(Data)
# : dict[str, list[Data]
def data_dict_to_descriptor_dict(model, device, data_dict, desc="Evaluation", leave_tqdm=True):
    descriptor_dict = {}

    for key, data_list in tqdm(data_dict.items(), desc=desc, leave=leave_tqdm):
        desc_list = []
        for data in data_list:
            desc_list.append(single_data_to_descriptor(model, device, data.clone()))
        descriptor_dict[key] = desc_list
        #model.to(device)
    return descriptor_dict


#
# Helper functions
#

def distance(descriptor1, descriptor2, p_norm=2):
    return torch.dist(descriptor1, descriptor2, p=p_norm)

from dataclasses import dataclass
import dataclasses
@dataclass
class BaseMetric:
    tp: int
    fp: int
    tn: int
    fn: int

    @classmethod
    def from_instance(cls, instance):
        return cls(**dataclasses.asdict(instance))


@dataclass
class ScoreMetric(BaseMetric):
    accuracy: float = None
    precision: float = None
    recall: float = None
    f1: float = None
    FRR: float = None  # (false neg rate) FRR false reject
    FAR: float = None  # (false pos rate) FAR accept rate

    def __str__(self):
        return f"ScoreMetric(tp={self.tp}, fp={self.fp}, tn={self.tn}, fn={self.fn}, " + \
               f"acc={self.accuracy:.4f}, preci={self.precision:.3f} recall={self.recall:.3f}, f1={self.f1:.3f}, " + \
               f"FRR={self.FRR:.3f}, FAR={self.FAR:.3f})"

#
# Metrics
#


def compare_two_unique_desc_lists(margin: float, desc_list1, desc_list2):
    # for i in range(min(len(desc_list1), len(desc_list2))):  # Assert torch.all, torch.eq isnt working, doing it manually
    #     assert not torch.all(torch.eq(desc_list1[i], desc_list2[i]))
    within_margin = 0
    outside_margin = 0
    for desc1 in desc_list1:
        for desc2 in desc_list2:
            d = distance(desc1, desc2)
            if d < margin:
                within_margin += 1
            else:
                outside_margin += 1
    return within_margin, outside_margin


# Assumes dist(desc1, desc2) == dist(desc2, desc1)
def get_base_metric_all_vs_all(margin: float, descriptor_dict):
    metrics = BaseMetric(tp=0, fp=0, tn=0, fn=0)

    # Check all vs all, same ID
    for _, desc_list in descriptor_dict.items():
        idxs_done = []
        for idx1, desc1 in enumerate(desc_list):
            for idx2, desc2 in enumerate(desc_list):
                if idx2 in idxs_done or idx1 == idx2:  # Skip if already checked, or same id
                    continue
                d = distance(desc1, desc2)
                if d < margin:
                    metrics.tp += 1
                else:
                    metrics.fn += 1
            idxs_done.append(idx1)


        # rest_descriptors = desc_list[1:]  # A list containing elements which has not been used as a pivot (yet)
        # for desc1 in desc_list:
        #     for desc2 in rest_descriptors:
        #         d = distance(desc1, desc2)
        #         if d < margin:
        #             metrics.tp += 1
        #         else:
        #             metrics.fn += 1
        #     del rest_descriptors[:1]  # Delete the leftmost element, which was just used as a pivot

        # Same as:
        # for i in range(len(desc_list)):
        #     for j in range(i+1, len(desc_list)):  # From i+1 to the end (exclusive)
        #         d = distance(desc_list[i], desc_list[j])




    # Want to only calculate one way (desc1 vs desc2)
    ids_done = []  # Yes this is suboptimal, but has better readability
    for id1, desc1_list in descriptor_dict.items():
        for id2, desc2_list in descriptor_dict.items():
            if id2 in ids_done or id1 == id2:  # Skip if already checked, or same id
                continue
            within_margin, outside_margin = compare_two_unique_desc_lists(margin, desc1_list, desc2_list)
            # If they are within the margin, they are false positives, as the compare ID's are not the same
            metrics.fp += within_margin
            # If they are outside the margin, they are true negatives, as they are correctly identified as unique
            metrics.tn += outside_margin

        ids_done.append(id1)

    return metrics

# torch_geometric.utils.  accuracy, precision, recall, f1
def get_metric_all_vs_all(margin: float, descriptor_dict):
    base_metric = get_base_metric_all_vs_all(margin=margin, descriptor_dict=descriptor_dict)
    metrics = ScoreMetric.from_instance(base_metric)  # Inherit from the base metric

    epsilon = 0.000001
    def nonzero(maybe_zero):
        return max(maybe_zero, epsilon)

    tp = base_metric.tp
    tn = base_metric.tn
    fp = base_metric.fp
    fn = base_metric.fn

    metrics.accuracy = (tp + tn) / (tp + tn + fp + fn)
    metrics.precision = tp / nonzero(tp + fp)
    metrics.recall = tp / nonzero(tp + fn)
    metrics.f1 = 2 * (metrics.precision * metrics.recall) / nonzero(metrics.precision + metrics.recall)
    metrics.FRR = fn / nonzero(fn + tp)
    metrics.FAR = fp / nonzero(fp + tn)

    return metrics


