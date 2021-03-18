from torch_geometric.data import Data
import torch_geometric.utils
import torch
from tqdm import tqdm
import onlineTripletLoss

#
# Processing of data before metrics
#


@torch.no_grad()  # This function disables autograd, so no training can be done on the data
def single_data_to_descriptor(model, device, data):
    data.to(device)
    # Assuming single data, ergo dont need to do .to_data_list()
    descriptor = model(data)
    descriptor.to("cpu")  # Results are returned to the cpu, as ASAPooling with grad would use over 12gb memory
    return descriptor


# Assumes dict: ID -> name -> Data
# : dict[str, dict[str, [Data]]
def data_dict_to_descriptor_dict(model, device, data_dict, desc="Evaluation", leave_tqdm=True):
    descriptor_dict = {}

    for key, data_list in tqdm(data_dict.items(), desc=desc, leave=leave_tqdm):
        assert isinstance(data_list, dict)
        desc_dict = {}
        for name, data in data_list.items():  # Keep the file metatadata if needed later
            desc_dict[name] = single_data_to_descriptor(model, device, data.clone())

        descriptor_dict[key] = desc_dict
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


# TODO make into part of class?
def generate_score_metric_from_base(base_metric: BaseMetric):
    metrics = ScoreMetric.from_instance(base_metric)  # Inherit from the base metric

    epsilon = 0.000001

    def nonzero(maybe_zero):
        return max(maybe_zero, epsilon)

    tp = base_metric.tp
    tn = base_metric.tn
    fp = base_metric.fp
    fn = base_metric.fn

    metrics.accuracy = (tp + tn) / (tp + tn + fp + fn)
    metrics.precision = tp / nonzero(tp + fp)  # Also called Positive Predictive Value (PPV)
    metrics.recall = tp / nonzero(tp + fn)  # Also called True positive rate or sensitivity
    metrics.f1 = 2 * (metrics.precision * metrics.recall) / nonzero(metrics.precision + metrics.recall)
    metrics.FRR = fn / nonzero(tp + fn)  # Also called FNR False negative rate
    metrics.FAR = fp / nonzero(tn + tn)  # Also called FPR False positive rate
    # Specificity (SPC) or True Negative Rate (TNR):    # SPC = TN / N = TN / (FP + TN)
    # Negative Predictive Value (NPV):     # NPV = TN / (TN + FN)
    # Fall-Out or False Positive Rate (FPR):    # FPR = FP / N = FP / (FP + TN) =1 – SPC
    # False Discovery Rate (FDR):    # FDR = FP / (FP + TP) =1 – PPV  # Good pga ikke TN
    # Mathews Correlation Coefficient (MCC):    # MCC=(TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP) *(TN+FN))

    return metrics

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

    descriptors = []
    label_ident = []
    label_name = []

    # Collapse the dict to one list
    for ident, ident_dict in descriptor_dict.items():
        for name, data in ident_dict.items():
            tmp2 = name.split("_")[1]
            if "03" in tmp2 or "04" in tmp2:
                continue
            descriptors.append(data)
            label_ident.append(ident)
            label_name.append(name)

    assert len(descriptors) == len(label_ident) == len(label_name)
    descriptors = torch.stack(descriptors)
    distances = onlineTripletLoss.pairwise_distances(embeddings=descriptors)

    margins = [0.1, 0.5, 1, 1.5, 2, 3, 4]

    for margin in margins:
        metrics = BaseMetric(tp=0, fp=0, tn=0, fn=0)
        # TODO make in matrix notation?
        # Check all vs all, same ID
        length = descriptors.size()[0]
        for idx1 in range(length):
            for idx2 in range(idx1+1, length):  # Do not check it against previous checked (Half of matrix used only)
                assert idx1 != idx2  # Do not check against itself
                assert label_name[idx1] != label_name[idx2]  # Dont use label name for other than verification


                d = distances[idx1, idx2]
                if label_ident[idx1] == label_ident[idx2]:
                    # Same Label, different ident
                    if d < margin:  # Good
                        metrics.tp += 1
                    else:           # Bad
                        metrics.fn += 1
                else:
                    # Different label
                    if d >= margin:  # Good
                        metrics.tn += 1
                    else:            # Bad
                        metrics.fp += 1
        print(f"margin:{margin}, {generate_score_metric_from_base(metrics)}")

    return metrics


# torch_geometric.utils.  accuracy, precision, recall, f1
def get_metric_all_vs_all(margin: float, descriptor_dict):
    base_metric = get_base_metric_all_vs_all(margin=margin, descriptor_dict=descriptor_dict)
    del descriptor_dict
    return generate_score_metric_from_base(base_metric)


def get_metric_gallery_set_vs_probe_set_BU3DFE(gallery_descriptors, probe_descriptors):
    base_metric = None
    return generate_score_metric_from_base(base_metric)
