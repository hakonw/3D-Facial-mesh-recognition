import torch

# Functions used to generate triplets

def euclidean_distance(descriptor1, descriptor2):
    return torch.dist(descriptor1, descriptor2, p=2)

# Not the loss used in train, as it meant for 1 triplet at a time
def triplet_loss(margin, anchor, pos, neg):
    return max(euclidean_distance(anchor, pos) - euclidean_distance(anchor, neg) + margin, 0)

# TODO pairs as input is wrong (too restrictive), fix sometime
def findtriplets(pairs, req_distance=1.0, accept_all=True):
    assert len(pairs) > 1
    valid_triplets = []

    # Goal: Find the triplet where a neg is closer to an anchor than a pos, with min/max distance
    # Solution, only find hard triplets atm, dont care about maximizing hard triplets
    #  Also allow for semi-hard
    # THIS SHOULD BE DONE WITH A MASK AND PYTORCH, instead of this massive n^2 * m^3   ish behaviour
    # AS IT CAN BE DONE MORE EFFICIENT THAN FOR LOOPS

    for i, list1 in enumerate(pairs):
        for j, list2 in enumerate(pairs):
            if i != j:  # Dont check same id against each other
                best_loss = -1 if accept_all else 0  # Higher loss is better for training, set to 0 to only include non-zero losses
                best_triplet = None

                # Find the best triplet for list1
                for idx1, list1_element1 in enumerate(list1):
                    for idx2, list1_element2 in enumerate(list1):
                        if idx1 != idx2:
                            # Here, all possible combinations of the pos/neg is created
                            #   For 2 elements, they are (1,2), (2,1)
                            #   for 3 elements, they are (1,2,3), (1,3,2), (2,1,3), (2,3,1), (3,1,2), (3,2,1)
                            #   O(n!) behaviour
                            # Now check against all possible elements and find the best
                            for neg_sample in list2:
                                score = triplet_loss(req_distance, list1_element1, list1_element2, neg_sample)
                                if score > best_loss:
                                    best_loss = score
                                    best_triplet = (list1_element1, list1_element2, neg_sample)
                if best_triplet is not None:
                    # Sanity assert, reduces speed by 0.8 it/s
                    # assert any([(best_triplet[0] == c_).all() for c_ in list1])  # Anc
                    # assert any([(best_triplet[1] == c_).all() for c_ in list1])  # Pos
                    # assert any([(best_triplet[2] == c_).all() for c_ in list2])  # Neg
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
