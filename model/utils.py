import torch
from torch_geometric.data import Data

def yield_file(in_file):
    f = open(in_file)
    buf = f.read()
    f.close()
    for b in buf.split('\n'):
        if b.startswith('v '):
            yield ['v', [float(x) for x in b.split(" ")[1:]]]
        elif b.startswith('f '):
            triangles = b.split(' ')[1:]
            if triangles[-1] == '':
                triangles = triangles[:-1]
            # -1 as .obj is base 1 but the Data class expects base 0 indices
            yield ['f', [int(t.split("/")[0]) - 1 for t in triangles]]
        else:
            yield ['', ""]


def read_obj(in_file, triangulation=True):
    vertices = []
    faces = []

    for k, v in yield_file(in_file):
        if k == 'v':
            vertices.append(v)
        elif k == 'f':
            faces.append(v)

    if not len(faces) or not len(vertices):
        return None

    if triangulation:
        assert len(faces[0]) >= 3
        faces_tri = []
        for face in faces:
            for i in range(1, len(face)-1):  # For quad it is from 1 -> 4-1 (not included)
                tri_corner0 = face[0]
                tri_corner1 = face[i]
                tri_corner2 = face[i+1]
                faces_tri.append([tri_corner0, tri_corner1, tri_corner2])
        faces = faces_tri

    pos = torch.tensor(vertices, dtype=torch.float)
    face = torch.tensor(faces, dtype=torch.long).t().contiguous()

    data = Data(pos=pos, face=face)

    return data

def euclidean_distance(descriptor1, descriptor2):
    return torch.dist(descriptor1, descriptor2, p=2)

def triplet_loss(margin, anchor, pos, neg):
    return max(euclidean_distance(anchor, pos) - euclidean_distance(anchor, neg) + margin, 0)

def findpairs(pairs, req_distance=1.0, accept_all=True):
    assert len(pairs) > 1
    valid_triplets = []

    # Goal: Find the triplet where a neg is closer to an anchor than a pos, with min/max distance
    # Solution, only find hard triplets atm, dont care about maximizing hard triplets
    #  Also allow for semi-hard
    # THIS SHOULD BE DONE WITH A MASK AND PYTORCH, instead of this massive n^2 * m^3   ish behaviour

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
