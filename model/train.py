import torch
from tqdm import tqdm

import datasetFacegen
# from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader
import utils
import tripletutils
import metrics

import dataclasses
import os

import onlineTripletLoss

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    print("MISSING CUDA")
    raise Exception("Missing cuda")

print("Cuda:", torch.cuda.is_available())
print("Type:", device.type)

from setup import Config

def train(cfg: Config):
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    import numpy as np
    np.random.seed(1)
    import random
    random.seed(1)
    # torch.set_deterministic(True)  # Currently impossible

    writer = cfg.WRITER

    # Actual code
    print("Loading model")
    model = cfg.MODEL
    model = model.to(device)

    # Loss & optimizer
    print("Loading criterion and optimizer")
    criterion = torch.nn.TripletMarginLoss(margin=cfg.MARGIN, p=cfg.P, reduction=cfg.REDUCTION)  # https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html#torch.nn.TripletMarginLoss   mean or sum reduction possible
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.LR)

    print("Loading DataLoader")
    # Note custom collate fn
    dataloader = DataLoader(dataset=cfg.DATASET, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, collate_fn=utils.list_collate_fn, drop_last=True)

    print("Staring")
    iter = 0
    for epoch in range(cfg.EPOCHS):
        tq = tqdm(enumerate(dataloader), desc=f"Epoch {epoch}/{cfg.EPOCHS}", leave=cfg.LEAVE_TQDM)

        losses = []
        for i_batch, batch in tq:
            optimizer.zero_grad()

            # Sample_batched is a list, instead of a pytorch_geometric batched object
            # This disables batching on the GPU
            # This is a feature that should be implemented if this code is used fore more than experimenting

            # batched is also a list of variable sized elements, where each list is a separate identity

            # TODO figure out what to do with dict
            # Ignore the metadata (names) of the idents
            assert(isinstance(batch[0], dict))
            batch = [list(b.values()) for b in batch]

            descritors = []
            for i in range(len(batch)):
                ident = [model(d.to(device)) for d in batch[i]]
                descritors.append(ident)

            # Første id: anchor og pos
            # Alle andre: Negative
            negs = []
            for i in range(1, len(descritors)):
                for m in descritors[i]:
                    negs.append(m)
            anc = descritors[0][1].to("cpu").unsqueeze(0).expand(len(negs), -1)
            pos = descritors[0][0].to("cpu").unsqueeze(0).expand(len(negs), -1)
            negs = torch.stack(negs).to("cpu")
            loss = criterion(anc, pos, negs)


            # anchors, positives, negatives = tripletutils.findtriplets(descritors, accept_all=cfg.ALL_TRIPLETS)
            # loss = criterion(anchors, positives, negatives)

            # # Unwrap from list for each ident, to a single long list with all
            # all = []
            # labels = []  # indencies for all, eks [0,0,0,1,1,2,2,3,3]
            # for ident, listt in enumerate(descritors):
            #     all += listt
            #     labels += [ident] * len(listt)
            #
            # all = torch.stack(all).to("cpu")
            # labels = torch.tensor(labels).to("cpu")
            # loss, fraction_positive_triplets = onlineTripletLoss.batch_all_triplet_loss(labels=labels, embeddings=all, margin=1.0)



            iter += 1
            losses.append(loss.item())
            if writer is not None:
                writer.add_scalar('Loss/train', loss.item(), iter)
                #writer.add_scalar('Pairs/train', len(anchors), iter)
            if iter % 5 == 0:
                tq.set_postfix(avg_loss=sum(losses)/max(len(losses), 1) ) #, fraction=fraction_positive_triplets.item()) #, pairs=len(anchors))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Metrics
        if (epoch + 1) % cfg.EPOCH_PER_METRIC == 0:
            descriptor_dict = metrics.data_dict_to_descriptor_dict(model=model, device=device, data_dict=cfg.DATASET_HELPER.get_cached_dataset(), desc="Evaluation/Test", leave_tqdm=False)
            metric = metrics.get_metric_all_vs_all(margin=1.0, descriptor_dict=descriptor_dict)
            print(metric)
            metric_dict = dataclasses.asdict(metric)
            for metric_key in metric_dict.keys():
                if writer is not None:
                    writer.add_scalar("metric-" + metric_key + "/train", metric_dict[metric_key], iter)

        if writer is not None:
            writer.add_scalar('AverageEpochLoss/train', sum(losses)/len(losses), epoch)
            writer.flush()

    print("Beginning metrics")
    descriptor_dict = metrics.data_dict_to_descriptor_dict(model=model, device=device, data_dict=cfg.DATASET_HELPER.get_cached_dataset())
    final_metrics = metrics.get_metric_all_vs_all(margin=1.0, descriptor_dict=descriptor_dict)
    print(final_metrics)

    # TODO the sub-section to own file

    # Create embeddings plot
    if (cfg.TENSORBOARD_EMBEDDINGS):
        labels = []
        features = []
        for id, desc_list in descriptor_dict.items():
            # TODO wrong if input to descriptor is dict
            for name, desc in desc_list.items():
                labels.append(f"{id}-{name}")
                features.append(desc)
        embeddigs = torch.stack(features)
        writer.add_embedding(mat=embeddigs, metadata=labels, tag=cfg.run_name)

    if (cfg.TENSORBOARD_HPARAMS):
        metric_dict = dataclasses.asdict(final_metrics)
        newdict = {}
        for metric_key in metric_dict.keys():
            newdict["hparam-" + metric_key + "/train"] = metric_dict[metric_key]
        writer.add_hparams(
            hparam_dict={
                "Epochs": cfg.EPOCHS,
                "Bsize": cfg.BATCH_SIZE,
                "Model": str(cfg.MODEL.__class__.__name__),
                "lr": cfg.LR,
                "Modelsummary": model.short_rep(),
              },
            metric_dict=dataclasses.asdict(final_metrics),
            run_name="hparam"
         )

    if cfg.WRITE_MODEL_SUMMARY:
        model_file = open(os.path.join(cfg.log_dir, cfg.run_name, "model.txt"), "x")
        model_file.write(str(model))
        model_file.close()

    if cfg.TENSORBOARD_TEXT:
        writer.add_text("Model-summary",
                        f"Run: {cfg.run_name}  \n" +
                        f"Result: {str(final_metrics)}  \n" +
                        "## Model  \n" +
                        str(model).replace("\n", "  \n"))

    # Close tensorboard
    if writer is not None:
        writer.close()

    return final_metrics


if __name__ == "__main__":
    train(Config())
