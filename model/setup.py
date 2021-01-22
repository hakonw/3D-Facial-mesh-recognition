import torch
import os
from torch.utils.tensorboard import SummaryWriter
import network


class Config:
    EPOCHS = 80
    BATCH_SIZE = 10
    NUM_WORKERS = 2

    EPOCH_PER_METRIC = 4

    MODEL = network.TestNet()

    # Loss function
    MARGIN = 1.0
    P = 2
    REDUCTION = "mean"  # mean or sum (or other, see pytorch doc)
    ALL_TRIPLETS = True  # To allow soft triplets (loss=0) & to have a comparable loss, or else have comparable triplets

    # Optimizer
    LR = 5e-4

    # Dataset
    DATASET_PATH = "/lhome/haakowar/Downloads/FaceGen_DB/"
    DATASET_SAVE = True
    DATASET_EDGE = True

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
    # WRITER.add_hparams(
    #     {  # hparam_dict
    #         "Epochs": EPOCHS,
    #         "Bsize": BATCH_SIZE,
    #         "Model": str(MODEL.__class__.__name__),
    #         "lr": LR
    #      },
    #     {}  # metric_dict
    # )
