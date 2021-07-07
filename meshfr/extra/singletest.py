import torch
import meshfr.train as train
import torch_geometric.transforms as T

from meshfr.io.read_wrl import read_wrl
from meshfr.io.sampler import data_all_sample, data_bruteforce_sample
from meshfr.io.read_bnt import read_bnt_raw
from meshfr.io.read_abs import read_abs_raw_gzip


with torch.no_grad():
    # Set CPU or CUDA 
    # If you run out of memory, I recommend CPU as it is only a single test
    device = torch.device('cpu')
    # device = torch.device('cuda')
    print("Running on", device)

    # Load model
    print("Loading feature extraction architecture")
    model = train.TestNet55_descv2().to(device)
    print("Loading feature siamese architecture")
    siam = train.Siamese_part().to(device)

    # Comment to use an untrained model
    print("Loading model checkpoint")
    loggingdir = "logging-siamese-1905-namechange-trash"
    experiment_folder = "2021-06-27_lr1e-03_batchsize50_frgc-norm-translate001-rotate5-axis012-siamesev2-fresh-full-net-with-loss-try3"
    model.load_state_dict(torch.load(f"./{loggingdir}/{experiment_folder}/model-5000.pt", map_location=device))
    siam.load_state_dict(torch.load(f"./{loggingdir}/{experiment_folder}/model-siam-5000.pt", map_location=device))


    # Preprocessing of dataset
    # These settings are ignored IF it finds a cached version (the .p file)
    sample_technique = "all"
    sample_size = 1024*12  # only used if sample technique is "bruteforce"
    assert sample_technique in ["bruteforce", "all"]
    # Load 3DFace dataset as a dict
    # If data is loaded manually, this part can be commented out
    from meshfr.datasets.dataset3DFace import get_3dface_dict
    d3face_path = "/lhome/haakowar/Downloads/3DFace_DB/3DFace_DB/"
    d3face_dict = get_3dface_dict(d3face_path, pickled=True, force=False, picke_name="/tmp/3dface-all.p", sample=sample_technique, sample_size=sample_size)


    # Select sample A and B
    #                       id   filename
    sampleA = d3face_dict["000"]["000_0"]
    sampleB = d3face_dict["002"]["002_0"]
    # # HERE, SAMPLES CAN MANUALLY BE LOADED LIKE
    # sampleA = torch_geometric.io.read_ply("./path/to/file.ply")
    # sampleA = read_wrl("./path/to/file.wrl")
    # # Use data_all_sample OR data_bruteforce_sample to get a data object with BNTs or ABSs files as they return the raw data
    # sampleA = data_all_sample(read_bnt_raw("./path/to/file.bnt"))
    # sampleA = data_all_sample(read_abs_raw_gzip("./path/to/file.abs.gz"))
    # sampleA = data_bruteforce_sample(read_bnt_raw("./path/to/file.bnt"), 2048)         # Has pre-sampling applied
    # sampleA = data_bruteforce_sample(read_abs_raw_gzip("./path/to/file.abs.gz"), 2048) # Has pre-sampling applied
    print("Sample A:", sampleA)
    print("Sample B:", sampleB)

    # Transfomration before using the feature extraction
    # FaceToEDge is required as it makes the mesh to a graph
    TRANSFORM = T.Compose([T.FaceToEdge(remove_faces=True), T.NormalizeScale()])

    print("Processing sample A and B")
    sampleA = TRANSFORM(sampleA).to(device)
    sampleB = TRANSFORM(sampleB).to(device)

    print("Sending samples though the model on", device)
    sampleA_features = model(sampleA)
    sampleB_features = model(sampleB)

    print("Running the samples though the siamese network")
    siamese_out = siam(sampleA_features, sampleB_features)

    print("--OUTPUT--")
    print("Raw siamese output:", siamese_out.item())

