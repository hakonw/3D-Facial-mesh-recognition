from numpy.lib.function_base import _parse_gufunc_signature
import meshfr.datasets.datasetFRGC as datasetFRGC
import torch
import meshfr.evaluation.metrics as metrics
import meshfr.testing as testing
from timeit import default_timer as timer
from timeit import timeit
from datetime import timedelta
import os
import torch_geometric.transforms as T
from meshfr.datasets.datasetGeneric import GenericDataset
from meshfr.datasets.datasetGeneric import DataLoader 
import torch_geometric.data.batch as geometric_batch


# Note: Timeit is used to get a more accurate time.
# The for loop is most for testing

with torch.no_grad():
    pickled = True
    force = False
    sample = "bruteforce" # bruteforce, 2pass, all, random
    sample_size = [1024*2, 1024*6][0]

    assert torch.cuda.is_available()
    # device = torch.device('cpu')
    device = torch.device('cuda')
    print(device)

    # Load model
    model = testing.TestNet55_descv2().to(device)
    siam = testing.Siamese_part().to(device)
    print("Loading save")
    # model.load_state_dict(torch.load("./logging-siamese-1905-namechange-trash/2021-06-13_lr1e-03_batchsize10_testing-bu3dfe-norm-translate001-rotate5-axis012/model-6000.pt", map_location=device))
    # siam.load_state_dict(torch.load("./logging-siamese-1905-namechange-trash/2021-06-13_lr1e-03_batchsize10_testing-bu3dfe-norm-translate001-rotate5-axis012/model-siam-6000.pt", map_location=device))

    POST_TRANSFORM = T.Compose([T.FaceToEdge(remove_faces=True), T.NormalizeScale()])

    # Load dataset
    frgc_path = "/lhome/haakowar/Downloads/FRGCv2/Data/"
    datadict_frgc_fall_2003 = datasetFRGC.get_frgc_dict(frgc_path + "Fall2003range", pickled=pickled, force=force, picke_name="/tmp/FRGCv2-fall2003_cache-2048-new.p", sample=sample, sample_size=sample_size)
    datadict_frgc_spring_2004 = datasetFRGC.get_frgc_dict(frgc_path + "Spring2004range", pickled=pickled, force=force, picke_name="/tmp/FRGCv2-spring2004_cache-2048-new.p", sample=sample, sample_size=sample_size)
    
    # Combine 
    combined = {}
    for ident, scans in datadict_frgc_fall_2003.items():
        if ident not in combined:
            combined[ident] = {}
        for s, v in scans.items():
            combined[ident][s] = v
    for ident, scans in datadict_frgc_spring_2004.items():
        if ident not in combined:
            combined[ident] = {}
        for s, v in scans.items():
            combined[ident][s] = v

    gallery_dict_data = {}
    probe_dict_data = {}
    # Remove all uneeded samples 
    for ident, scans in combined.items():
        if ident not in gallery_dict_data: gallery_dict_data[ident] = {}
        if ident not in probe_dict_data: probe_dict_data[ident] = {}
        ident_dict_items_sorted = sorted(scans.items(), key=lambda item: int(item[0].split("d")[1]))
        assert int(ident_dict_items_sorted[0][0].split("d")[1]) == min([int(it[0].split("d")[1]) for it in ident_dict_items_sorted])
        if len(ident_dict_items_sorted) > 1:
            assert int(ident_dict_items_sorted[0][0].split("d")[1]) < int(ident_dict_items_sorted[1][0].split("d")[1])

        # First ident in gallery
        first_ident = ident_dict_items_sorted.pop(0)
        gallery_dict_data[ident][first_ident[0]] = first_ident[1]

        # Rest in probe
        for name, data in ident_dict_items_sorted:
            probe_dict_data[ident][name] = data
    assert len(gallery_dict_data) == 466
    
    gallery_dataloader = GenericDataset(gallery_dict_data, POST_TRANSFORM)
    dataloader = DataLoader(gallery_dataloader, batch_size=100, shuffle=False, num_workers=0, drop_last=False)

    # Simple sample    
    sample = datadict_frgc_fall_2003["02463"]["02463d546"]


    print("Pre-processing")
    t = timeit(lambda: POST_TRANSFORM(sample), number=1000)
    print(f"  tiemit: {t/1000:f}")
    # See manually, will be worse
    for i in range(4):
        start = timer()
        out = POST_TRANSFORM(sample)
        end = timer()
        print(" ", timedelta(seconds=end-start))


    # Single sample:
    sample = POST_TRANSFORM(sample).to(device)
    print("single model pass (Feature extraction)")
    t = timeit(lambda: model(sample), number=100)
    print(f"  tiemit: {t/100:f}")
    for i in range(5):
        start = timer()
        out = model(sample)
        end = timer()
        print(" ", timedelta(seconds=end-start))
    
    # sample = model(sample)
    # Fetch a random probe (the first avaiable)
    id = list(probe_dict_data.keys())[0]
    scan = list(probe_dict_data[id].keys())[0]
    sample = model(POST_TRANSFORM(probe_dict_data[id][scan]).to(device))

    # To test if the time is constant, use custom function
    @torch.no_grad()
    def generate_descriptor_dict_from_dataloader(model, dataloader, device):
        model.eval()
        descriptor_dict = {}
        for larger_batch in dataloader:  # contains a list with each batch
            datas = []
            for b in larger_batch:  # Collapse list to a single large batch object
                datas += b.to(device).to_data_list()
            batch = geometric_batch.Batch.from_data_list(datas).to(device)
            i = timer()
            output = model(batch)
            o = timer()
            print(" length:", len(datas), "time:", timedelta(seconds=o-i))
            t = timeit(lambda: model(batch), number=100)
            print(f"  tiemit: {t/100:f}")

            ids = batch.dataset_id  # Real ID, not generated number id
            names = batch.name
            for idx, id in enumerate(ids):
                if id not in descriptor_dict:
                    descriptor_dict[id] = {}
                descriptor_dict[id][names[idx]] = output[idx]

        return descriptor_dict

    # Preload the gallery descriptors
    print("Preloading gallery descriptors")
    print("printing per (larger batch) time")
    gallery_dict = generate_descriptor_dict_from_dataloader(model=model, dataloader=dataloader, device=device)
    assert len(gallery_dict) == 466

    print("Flattening")
    gal_datas, gal_labels = metrics.generate_flat_descriptor_dict(gallery_dict, device)
    assert len(gal_datas) == 466
    assert len(gal_labels) == 466


    # Create a matrix to pass to siamese
    # matrix containing all [1, len(gals)]
    print("gal_datas.shape", gal_datas.shape)  # [466, 128]
    print("sample.shape", sample.shape) # [1, 128]

    part1 = gal_datas

    print("Expand")
    t = timeit(lambda: sample.expand(gal_datas.shape[0], -1), number=100)
    print(f"  tiemit: {t/100:f}")
    for i in range(4):
        start = timer()
        part2 = sample.expand(gal_datas.shape[0], -1)
        end = timer()
        print(" ", timedelta(seconds=end-start))
    part2 = sample.expand(gal_datas.shape[0], -1)
    print("part2.shape", part2.shape)

    part1 = part1.to(device)
    part2 = part2.to(device)

    print("Siamese")
    t = timeit(lambda: siam(part1, part2), number=100)
    print(f"  tiemit: {t/100:f}")
    for i in range(5):
        start = timer()
        result_matrix = siam(part1, part2)
        end = timer()
        print(" ", timedelta(seconds=end-start))
    
    result_matrix = siam(part1, part2)


    print("Matching")
    t = timeit(lambda: torch.argmax(result_matrix, dim=-1), number=100)
    print(f"  tiemit: {t/100:f}")
    for i in range(3):
        start = timer()
        torch.argmax(result_matrix, dim=-1)
        end = timer()
        print(" ", timedelta(seconds=end-start))

    print("result_matrix.shape", result_matrix.shape)
    idxs_max = torch.max(result_matrix, dim=-1)  # "Check each row"
    print(idxs_max)


    ### Extra 
    filename = "./bs104_N_N_3.bnt"
    import meshfr.io.sampler as sampler 
    from meshfr.io.read_bnt import read_bnt_raw
    import numpy as np 
    from scipy.spatial import Delaunay

    data = read_bnt_raw(filename)
    print("Time for pre-pre-processing")
    t = timeit(lambda: np.random.choice(data.shape[0], size=1024, replace=False), number=10000)
    print(f"  Simple sampling tiemit: {t/10000:f}")
    random_indices = np.random.choice(data.shape[0], size=1024, replace=False)
    reduced_data = data[random_indices, :]

    t = timeit(lambda: Delaunay(points=reduced_data[:, 0:2]), number=1000)
    print(f"  triangulation timeit: {t/1000:f}")
