import torch
import os
from torch.utils.tensorboard import SummaryWriter
import network
import datasetFacegen
import datasetBU3DFE


class Datasets:
    DATASET_SAVE = True
    DATASET_EDGE = True

    # Facegen Dataset
    @staticmethod
    def get_facegen_dataset():
        DATASET_PATH_FACEGEN = "/lhome/haakowar/Downloads/FaceGen_DB/"
        print("Dataset: Facegen")
        FACEGEN_HELPER = datasetFacegen.FaceGenDatasetHelper(root=DATASET_PATH_FACEGEN, pickled=Datasets.DATASET_SAVE, face_to_edge=Datasets.DATASET_EDGE)
        DATASET_FACEGEN = datasetFacegen.FaceGenDataset(FACEGEN_HELPER.get_cached_dataset())
        return FACEGEN_HELPER, DATASET_FACEGEN

    # BU-3DFE Dataset
    @staticmethod
    def get_bu3dfe_dataset():
        DATASET_PATH_BU3DFE = "/lhome/haakowar/Downloads/BU_3DFE/"
        print("Dataset: BU-3DGE")
        BU3DFE_HELPER = datasetBU3DFE.BU3DFEDatasetHelper(root=DATASET_PATH_BU3DFE, pickled=Datasets.DATASET_SAVE)
        DATASET_BU3DGE = datasetBU3DFE.BU3DFEDataset(BU3DFE_HELPER.get_cached_dataset())
        return BU3DFE_HELPER, DATASET_BU3DGE


class Config:
    # General
    EPOCHS = 80
    BATCH_SIZE = 4  # Note, currently the triplet selector is n^2 * m^2, or n^2 if n >> m (batch size vs scans per id)

    # Metrics
    EPOCH_PER_METRIC = 10

    # Model
    MODEL = network.TestNet2()  # TestNet (new) or PrelimNet (old)

    # Loss function
    MARGIN = 1.0
    P = 2
    REDUCTION = "mean"  # mean or sum (or other, see pytorch doc)

    # Triplet selector
    ALL_TRIPLETS = True  # To allow soft triplets (loss=0) & to have a comparable loss, or else have comparable triplets

    # Optimizer
    LR = 5e-4

    # Dataset and Dataloader
    NUM_WORKERS = 1  # for the dataloader. As it is in memory, a high number is not needed
    # DATASET = Datasets.DATASET_FACEGEN
    # DATASET_HELPER = Datasets.FACEGEN_HELPER
    # DATASET = Datasets.DATASET_BU3DGE
    # DATASET_HELPER = Datasets.BU3DFE_HELPER
    DATASET_HELPER, DATASET = Datasets.get_facegen_dataset()
    #  DATASET_HELPER, DATASET = Datasets.get_bu3dfe_dataset()

    # Various logger
    LEAVE_TQDM = True
    log_dir = "log/"

    TENSORBOARD_EMBEDDINGS = False
    TENSORBOARD_HPARAMS = True
    TENSORBOARD_TEXT = True
    WRITE_MODEL_SUMMARY = True

    def __init__(self, enable_writer=True):
        # Logger
        # https://pytorch.org/docs/stable/tensorboard.html
        if enable_writer:
            try:
                previous_runs = os.listdir(Config.log_dir)
                if len(previous_runs) == 0:
                    run_number = 1
                else:
                    previous_runs = list(filter(lambda x: "run" in x, previous_runs))
                    run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1
            except FileNotFoundError:
                print("File not found, defaulting run number")
                run_number = 0
            self.run_name = f"run_{run_number:03}"
            self.WRITER = SummaryWriter(log_dir=os.path.join(Config.log_dir, self.run_name), max_queue=20)
            print("Beginning", self.run_name)
        else:
            import random
            self.run_name = "No-log-UNKNOWN" + str(random.randint(0, 10000))
            self.WRITER = None
