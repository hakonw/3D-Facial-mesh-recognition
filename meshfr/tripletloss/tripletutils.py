import torch
from . import onlineTripletLoss


# Functions used to generate triplets
# TODO pairs as input is wrong (too restrictive), fix sometime
# TODO also replace with the onlinetripletloss?
def findtriplets(pairs, req_distance, accept_all=True):
    assert len(pairs) > 1
    valid_triplets = []

    # Unwrap from list for each ident, to a single long list with all
    all = []
    labels = []  # indencies for all, eks [0,0,0,1,1,2,2,3,3]
    for ident, list in enumerate(pairs):
        all += list
        labels += [ident]*len(list)

    all = torch.stack(all)
    labels = torch.tensor(labels)
    distances = onlineTripletLoss.pairwise_distances(all)  # Assume symmetric, a -> b
    mask = onlineTripletLoss.get_triplet_mask(labels)

    size = len(all)
    for a in range(size):
        for n in range(size):
            best_loss = -1 if accept_all else 0  # Higher loss is "better", set to 0 to only include non-zero losses
            best_triplet = None

            for p in range(size):
                if mask[a, p, n].all():
                    score = max(distances[a, p] - distances[a, n] + req_distance, 0)
                    if score > best_loss:
                        best_loss = score
                        best_triplet = (all[a], all[p], all[n])
            if best_triplet is not None:
                valid_triplets.append(best_triplet)

    # We should hopefully here have lots of good hard/semi-hard pairs
    # If there are none, create a negative triplet to not crash the loss function
    if len(valid_triplets) == 0:
        anchors = torch.stack([pairs[0][0]])
        positives = torch.stack([pairs[0][1]])
        negatives = torch.stack([pairs[1][0]])
        print("Filling triplets due to none valid")
        return anchors, positives, negatives

    anchors = []
    positives = []
    negatives = []
    for triplet in valid_triplets:
        anchors.append(triplet[0])
        positives.append(triplet[1])
        negatives.append(triplet[2])
    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    negatives = torch.stack(negatives)
    return anchors, positives, negatives
