import torch
import os
from torch.utils.tensorboard import SummaryWriter
import network


class Config:
    # General
    EPOCHS = 10
    BATCH_SIZE = 10  # Note, currently the triplet selector is n^2 * m^2, or n^2 if n >> m (batch size vs scans per id)

    # Metrics
    EPOCH_PER_METRIC = 4

    # Model
    MODEL = network.TestNet()  # TestNet (new) or PrelimNet (old)

    # Loss function
    MARGIN = 1.0
    P = 2
    REDUCTION = "mean"  # mean or sum (or other, see pytorch doc)

    # Triplet selector
    ALL_TRIPLETS = True  # To allow soft triplets (loss=0) & to have a comparable loss, or else have comparable triplets

    # Optimizer
    LR = 1e-4

    # Dataset
    NUM_WORKERS = 2  # for the dataloader
    DATASET_PATH = "/lhome/haakowar/Downloads/FaceGen_DB/"
    DATASET_SAVE = True
    DATASET_EDGE = True

    #
    TENSORBOARD_EMBEDDINGS = False
    TENSORBOARD_HPARAMS = True

    # Logger
    # https://pytorch.org/docs/stable/tensorboard.html
    try:
        previous_runs = os.listdir('log/')
        if len(previous_runs) == 0:
            run_number = 1
        else:
            run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1
    except FileNotFoundError:
        run_number = 0
    run_name = f"run_{run_number:03}"
    WRITER = SummaryWriter(log_dir=os.path.join("log", run_name), max_queue=20)
    print("Beginning", run_name)
