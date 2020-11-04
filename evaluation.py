import torch
from tqdm import tqdm
from pytorch3d.structures import Meshes

class Evaluation():
    def __init__(self, dataset, margin, device):
        #self.dataset = dataset
        self.margin = margin
        self.device = device

        self.dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                      batch_size=10,
                                                      shuffle=False,
                                                      num_workers=4,
                                                      drop_last=False)

    @staticmethod
    def distance(descriptor1, descriptor2):
        return torch.dist(descriptor1, descriptor2, p=2)  # TODO validate that euclidian is p-norm=2

    def __call__(self, model):
        # Structure the dataset:

        descriptor_dict = {}
        eval_index = 0

        tq = tqdm(enumerate(self.dataloader), desc=f"Evaluation", leave=False)
        for i_batch, sample_batced in tq:
            verts_reg = sample_batced["regular"]["verts"]
            verts_reg_idx = sample_batced["regular"]["verts_idx"]
            verts_alt = sample_batced["alt"]["verts"]
            verts_alt_idx = sample_batced["alt"]["verts_idx"]
            idd = sample_batced["idd"]

            verts = torch.cat([verts_reg, verts_alt], dim=0)
            verts_idx = torch.cat([verts_reg_idx, verts_alt_idx], dim=0)
            meshes = Meshes(verts=verts, faces=verts_idx)
            meshes.to(self.device)

            y_id = [int(i) for i in sample_batced["idd"] + sample_batced["idd"]]
            #y_hat = torch.tensor(y_hat_py, dtype=torch.long, device=device)

            y_pred = model(meshes)

            for i in range(len(y_pred)):
                descriptor_dict[eval_index] = {"descriptor":y_pred[i], "id":y_id[i]}
                eval_index += 1

        tp = 0
        fp = 0
        tn = 0
        fn = 0

        # Test everyone vs everyone
        for model1_id, id1_prediction in descriptor_dict.items():
            for model2_id, id2_prediction in descriptor_dict.items():
                id1 = id1_prediction["id"]
                id2 = id2_prediction["id"]
                desc1 = id1_prediction["descriptor"]
                desc2 = id2_prediction["descriptor"]

                if model1_id == model2_id:  # No need to check model
                    continue
                else:
                    d = Evaluation.distance(desc1, desc2)
                    d_same = d < self.margin

                    if id1 == id2:  # Same face, but different model
                        if d_same:
                            tp += 1  # good
                        else:
                            fn += 1  # Bad
                    if id1 != id2:  # Different face and model
                        if d_same:
                            fp += 1  # Bad
                        else:
                            tn += 1  # Good
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        accuracy = (tp+tn)/(tp+tn+fp+fn) # TODO imbalanced?
        print("Evaluation")
        print(f"  Precision: {precision}\n  Recall:    {recall}\n  Accuracy:  {accuracy}")
        print(f"  tp:{tp}, fp:{fp}, tn:{tn}, fn:{fn}")


