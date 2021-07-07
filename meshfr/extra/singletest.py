import torch
import meshfr.train as train
import torch_geometric.transforms as T

with torch.no_grad():
    # device = torch.device('cpu')
    device = torch.device('cuda')
    print(device)

    # Load model
    print("Loading feature extraction")
    model = train.TestNet55_descv2().to(device)
    print("Loading feature siamese")
    siam = train.Siamese_part().to(device)

    print("Loading save")
    # UNCOMMENT TO LOAD SAVE
    # loggingdir = "logging-siamese-1905-namechange-trash"
    # experiment = "2021-06-13_lr1e-03_batchsize10_testing-bu3dfe-norm-translate001-rotate5-axis012"
    # model.load_state_dict(torch.load(f"./{loggingdir}/{experiment}/model-6000.pt", map_location=device))
    # siam.load_state_dict(torch.load(f"./{loggingdir}/{experiment}/model-siam-6000.pt", map_location=device))


    # Transfomration before loading into the model
    POST_TRANSFORM = T.Compose([T.FaceToEdge(remove_faces=True), T.NormalizeScale()])

    # Load 3DFace dataset as a dict
    from meshfr.datasets.dataset3DFace import get_3dface_dict
    d3face_path = "/lhome/haakowar/Downloads/3DFace_DB/3DFace_DB/"

    # Preprocessing

    allowed_sample_techniques = ["bruteforce", "all"]
    sample_size = 4096*12  # only used if sample technique is "bruteforce"
    sample_technique = "all"
    assert sample_technique in allowed_sample_techniques

    d3face_dict = get_3dface_dict(d3face_path, pickled=True, force=False, picke_name="/tmp/3dface-12k.p", sample=sample_technique, sample_size=sample_size)

    # Simple sample
    sampleA = d3face_dict["000"]["000_0"]
    sampleB = d3face_dict["000"]["000_1"]
    # HERE SAMPLES CAN MANUALLY BE LOADED LIKE
    # sampleA = torch_geometric.io.read_ply("./path/to/file.ply")

    print("Processing sample A and B")
    sampleA = POST_TRANSFORM(sampleA).to(device)
    sampleB = POST_TRANSFORM(sampleB).to(device)

    print("Sending samples though the model")
    sampleA_feature_extraction = model(sampleA)
    sampleB_feature_extraction = model(sampleB)

    siamese_out = siam(sampleA, sampleB)

    print("--OUTPUT--")
    print("Raw siamese output:", siamese_out.item())

