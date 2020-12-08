import torch
from tqdm import tqdm
from pytorch3d.structures import Meshes

class Evaluation():
    def __init__(self, dataset, margin, device):
        #self.dataset = dataset
        self.margin = margin
        self.device = device
        self.batch_size=10
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                      batch_size=self.batch_size,
                                                      shuffle=False,
                                                      num_workers=4,
                                                      drop_last=False)

    @staticmethod
    def distance(descriptor1, descriptor2):
        return torch.dist(descriptor1, descriptor2, p=2)  # TODO validate that euclidian is p-norm=2

    def __call__(self, model, n_vs_nn=False):
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
                descriptor_dict[eval_index] = {"descriptor":y_pred[i], "id":y_id[i], "reg": i<self.batch_size}
                eval_index += 1

        tp = 0
        fp = 0
        tn = 0
        fn = 0

        models_done = []

        # Test everyone vs everyone
        for model1_id, id1_prediction in descriptor_dict.items():
            for model2_id, id2_prediction in descriptor_dict.items():
                id1 = id1_prediction["id"]
                id2 = id2_prediction["id"]
                desc1 = id1_prediction["descriptor"]
                desc2 = id2_prediction["descriptor"]

                # Skip duplicate results
                if model2_id in models_done:
                    continue

                if n_vs_nn:
                    # Restrict gal to be reg, probe to be alt
                    if not (id1_prediction["reg"] and not id2_prediction["reg"]):
                         continue

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
            # For each model1
            models_done.append(model1_id)

        precision = tp/(tp+fp)  # also called positive predictive value
        recall = tp/(tp+fn)  # also called true positive rate or 1 - false neg rate
        accuracy = (tp+tn)/(tp+tn+fp+fn) # TODO imbalanced?
        print("Evaluation")
        print(f"  Precision: {precision}\n  Recall:    {recall}\n  Accuracy:  {accuracy}")
        print(f"  (false neg rate)FRR false reject: {fn/(fn+tp)}, (false pos rate)FAR false accept: {fp/(fp+tn)}")
        print(f"  tp:{tp}, fp:{fp}, tn:{tn}, fn:{fn}")

        rank1_true_amount = 0
        rank1_false_amount = 0

        # For each probe, in for each img in gallery
        for anchor_self_id, anchor_prediction in descriptor_dict.items():
            anchor_id = anchor_prediction["id"]
            anchor_desc = anchor_prediction["descriptor"]

            if n_vs_nn:
                if anchor_prediction["reg"]:  # If probe is reg, slip it
                    continue

            best_id = -1  # default value
            best_dist = 10000000  # Inf ish

            for sample_self_id, sample_prediction in descriptor_dict.items():
                sample_id = sample_prediction["id"]
                sample_desc = sample_prediction["descriptor"]

                if n_vs_nn:
                    if not sample_prediction["reg"]:  # If gal is alt (non-reg) , slip it
                        continue

                if anchor_self_id == sample_self_id:  # Checking against itself, skip
                    continue
                else:
                    d = Evaluation.distance(anchor_desc, sample_desc)
                    if d < best_dist:
                        best_dist = d
                        best_id = sample_id  # Model id, not self id
            if anchor_id == best_id:
                rank1_true_amount += 1
            else:
                rank1_false_amount += 1
        print("Rank 1 evaul")
        rank1_acc = rank1_true_amount/(rank1_false_amount+rank1_true_amount)
        print(f"  true: {rank1_true_amount}, false: {rank1_false_amount}, rank1acc: {rank1_acc}")
