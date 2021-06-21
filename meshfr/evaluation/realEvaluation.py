import meshfr.evaluation.metrics as metrics
import matplotlib.pyplot as plt

markers = ["o", "^", "p", "X"]
colors = ["darkorange", "orangered", "royalblue", "limegreen"]



# Generic CMC
def generate_cmc_from_lists(ir_ranks, labels):
    def plotit(array, marker, color, label):
        plt.plot(list(range(1, len(array)+1)), array, marker=marker, lw=2, color=color, label=label, clip_on=False)
    
    assert len(ir_ranks) == len(labels)

    fig = plt.figure()
    plt.title("Cumulative Match Curve")
    for i in range(len(ir_ranks)):
        plotit(ir_ranks[i], markers[i], colors[i], labels[i])
    plt.ylabel("Identification Rate")
    plt.xlabel("Rank")
    plt.legend(loc="lower right")
    max_x = max([len(ar) for ar in ir_ranks])
    plt.xlim(left=1, right=max_x)
    plt.ylim(top=1.0005)
    plt.grid(True)
    plt.locator_params(axis="x", integer=True)
    # plt.savefig(os.path.join(savepath, f"bu3dfe-cmc-combined-{epoch}.pdf"), bbox_inches='tight')
    return fig

# BU-3DFE

import sklearn.metrics

def bu3dfe_filter_low_intensity(name):
    expression_scale = name.split("_")[1]
    return "01" in expression_scale or "02" in expression_scale
def bu3dfe_filter_high_intensity(name):
    expression_scale = name.split("_")[1]
    return "03" in expression_scale or "04" in expression_scale
def bu3dfe_generate_cmc(descriptor_dict, siam, device):
    gallery_dict, probe_dict = metrics.split_gallery_set_vs_probe_set_BU3DFE(descriptor_dict)

    # Assert gallery is correctly layed out
    for ident, ident_dict in gallery_dict.items():
        assert len(ident_dict) == 1

    def get_ir_by_rank(probe):
        rank_tp_dict  = metrics.generate_metirc_siamese_rank1_cmc(siam, device, gallery_dict, probe)
        ir_by_rank = metrics.generate_identification_rate_by_rank(rank_tp_dict)
        return ir_by_rank
    
    # N vs low
    probe_dict_low =  metrics.filter_descriptordict_scans(probe_dict, bu3dfe_filter_low_intensity)
    ir_by_rank_low = get_ir_by_rank(probe_dict_low)

    # N vs high
    probe_dict_high =  metrics.filter_descriptordict_scans(probe_dict, bu3dfe_filter_high_intensity)
    ir_by_rank_high = get_ir_by_rank(probe_dict_high)

    # N vs ALL
    ir_by_rank_all = get_ir_by_rank(probe_dict)
    
    ir_ranks = [ir_by_rank_low, ir_by_rank_high, ir_by_rank_all]
    labels = ["Neutral vs. Low-Intensity", "Neutral vs. High-Intensity", "Neutral vs. All"]
    return generate_cmc_from_lists(ir_ranks, labels)

# Bosp

bosphorus_filter_neutral = lambda name: "_N_N_" in name
bosphorus_filter_non_neutral = lambda name: "_N_N_" not in name
def bosphorus_generate_cmc(descriptor_dict, siam, device):
    gallery_dict, probe_dict = metrics.split_gallery_set_vs_probe_set_bosphorus(descriptor_dict)

    # Assert gallery is correctly layed out
    for ident, ident_dict in gallery_dict.items():
        assert len(ident_dict) == 1

    def get_ir_by_rank(probe):
        rank_tp_dict  = metrics.generate_metirc_siamese_rank1_cmc(siam, device, gallery_dict, probe)
        ir_by_rank = metrics.generate_identification_rate_by_rank(rank_tp_dict)
        return ir_by_rank

    # N vs low
    probe_dict_neut =  metrics.filter_descriptordict_scans(probe_dict, bosphorus_filter_neutral)
    ir_by_rank_neut = get_ir_by_rank(probe_dict_neut)

    # N vs high
    probe_dict_non_neut =  metrics.filter_descriptordict_scans(probe_dict, bosphorus_filter_non_neutral)
    ir_by_rank_non_neut = get_ir_by_rank(probe_dict_non_neut)

    # N vs ALL
    ir_by_rank_all = get_ir_by_rank(probe_dict)

    ir_ranks = [ir_by_rank_neut, ir_by_rank_non_neut, ir_by_rank_all]
    labels = ["Neutral vs. Neutral", "Neutral vs. Non-neutral", "Neutral vs. All"]
    return generate_cmc_from_lists(ir_ranks, labels)

# FRGC

def frgc_generate_cmc(descriptor_dict, siam, device):
    gallery_dict, probe_dict = metrics.split_gallery_set_vs_probe_set_frgc(descriptor_dict)

    # Assert gallery is correctly layed out
    for ident, ident_dict in gallery_dict.items():
        assert len(ident_dict) == 1

    # Cannot filter probe as there is no markings on what type it is 

    rank_tp_dict  = metrics.generate_metirc_siamese_rank1_cmc(siam, device, gallery_dict, probe_dict)
    ir_by_rank_all = metrics.generate_identification_rate_by_rank(rank_tp_dict)

    ir_ranks = [ir_by_rank_all]
    labels = ["First vs. Rest"]
    return generate_cmc_from_lists(ir_ranks, labels)


from sklearn.metrics import roc_curve, auc
import numpy as np
# ROC
def generate_roc_aucs(y_trues, y_scores):
    assert len(y_trues) == len(y_scores)
    false_positive_rates = []
    recalls = []
    roc_aucs = []

    for i in range(len(y_trues)):
        false_positive_rate, recall, thresholds = roc_curve(y_trues[i].cpu(), y_scores[i].cpu())
        roc_auc = auc(false_positive_rate, recall)
        false_positive_rates.append(false_positive_rate)
        recalls.append(recall)
        roc_aucs.append(roc_auc)
    return false_positive_rates, recalls, roc_aucs

def generate_roc_from_lists(false_positive_rates, recalls, roc_aucs, labels, log=False):
    def plotit(fpr, recall, roc_auc, marker, color, label):
        plt.plot(fpr, recall, lw=2, color=color, label=f"{label}, AUC = {roc_auc:.3f}", clip_on=True)   # No marker
    
    assert len(false_positive_rates) == len(recalls)
    assert len(false_positive_rates) == len(roc_aucs)
    assert len(false_positive_rates) == len(labels)
    z = 1e-10

    fig = plt.figure()
    plt.title("Receiver Operating Characteristic (ROC)")
    for i in range(len(false_positive_rates)):
        plotit(false_positive_rates[i], recalls[i], roc_aucs[i], markers[i], colors[i], labels[i])
    plt.legend(loc="lower right")
    # plt.legend(loc="upper left")
    x = np.linspace(0,1,101)
    plt.plot(x, x, lw=2, linestyle="--")  # RNG line which works for log too
    plt.ylabel("True Positive Rate (Recall)")
    plt.xlabel("False Acceptance Rate")
    plt.grid(True, which="both")
    if not log:
        plt.xlim([-0.05,1.05])
        plt.ylim([-0.05,1.05])
    if log:
        plt.xscale("log", nonpositive="clip")
        plt.xlabel("False Acceptance Rate (Log scale)")
        plt.xlim(left=1e-3, right=1.05)
        # plt.ylim([0.9,1.001])
    return fig



def generate_acc_fpr(false_positive_rates, accuracies, labels):
    assert len(false_positive_rates[0]) == len(accuracies[0])
    fig = plt.figure()
    plt.title("Validation Rate at different thresholds")
    assert len(false_positive_rates) == len(accuracies)
    assert len(false_positive_rates) == len(labels)
    for i in range(len(labels)):
        plt.plot(false_positive_rates[i], accuracies[i], lw=2, color=colors[i], label=f"{labels[i]}", clip_on=True)   # No marker
    plt.legend(loc="lower right")
    # plt.legend(loc="upper left")
    x = np.linspace(0,1,101)
    y = np.linspace(0.5,0.5,101)
    plt.plot(x, y, lw=2, linestyle="--")
    plt.ylabel("Validation Rate")
    plt.xlabel("False Acceptance Rate (Log scale)")
    plt.grid(True, which="both")
    plt.xscale("log", nonpositive="clip")
    plt.xlim(left=5e-5, right=1.05)
    plt.ylim(bottom=0.43, top=1.05)
    return fig

import inspect
def generate_verification_rates(y_trues, y_scores):
    accs, accuracies, fpr = [], [], []
    assert len(y_trues) == len(y_scores)
    for i in range(len(y_trues)):
        a1, a2, f = generate_veritication_rate(y_trues[i], y_scores[i])
        accs.append(a1)
        accuracies.append(a2)
        fpr.append(f)
    return accs, accuracies, fpr

def generate_veritication_rate(y_true, y_score, FAR_targets=[0.001, 0.01]): # TODO 0.001
    y_true = y_true.cpu()
    y_score = y_score.cpu()

    # Instead of doing it manually, use sklearns function
    #fpr, fnr, thresholds = det_curve(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_score) #   (tpr == recall)
    #precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    #fps, tps, thresholds2 = _binary_clf_curve(y_true, y_score) # Tresh is decreasing score values
    #print(thresholds)
    #print(thresholds2)
    
    #total_negative = fps[-1]
    #total_positive = tps[-1]
    total = y_true.shape[0]
    accs = []

    # Tresholds is decreasing, and fpr is increasing
    for FAR_target in FAR_targets:
        optimal_tresh = 1.0
        achived_fpr = 0.0
        next_fpr = 0.0
        for i in range(len(thresholds)):
            if fpr[i] > FAR_target:
                assert fpr[i-1] <= FAR_target
                optimal_tresh = thresholds[i-1]
                achived_fpr = fpr[i-1]
                next_fpr = fpr[i]
                break
        
        # Acc = TP + TN / total
        predicted = y_score>=optimal_tresh
        true_predicted = predicted.eq(y_true).sum()
        acc = true_predicted / total
        accs.append(acc)

        print(f"{inspect.stack()[2][3]}\tFAR: {FAR_target}, Accuracy: {acc:.5f} at tresh:{optimal_tresh:.5f} and fpr: {achived_fpr:.5f}, (prev: {next_fpr:.5f})")
        
        

    # Generate a curve for accuracy VS FPR
    # Will be as long as fpr, and tresholds
    accuracies = []
    for i in range(len(thresholds)):
        predicted = y_score >= thresholds[i]
        g_true_predicted = predicted.eq(y_true).sum()  # Both TP and TN
        accuracies.append(g_true_predicted/total)

    return accs, accuracies, fpr # , optimal_tresh, achived_fpr, next_fpr


def bu3dfe_generate_roc(descriptor_dict, siam, device):
    gallery_dict, probe_dict = metrics.split_gallery_set_vs_probe_set_BU3DFE(descriptor_dict)

    # Assert gallery is correctly layed out
    for ident, ident_dict in gallery_dict.items():
        assert len(ident_dict) == 1

    # N vs low
    probe_dict_low =  metrics.filter_descriptordict_scans(probe_dict, bu3dfe_filter_low_intensity)
    y_true_low, y_score_low  = metrics.generate_metric_siamese_roc_bal(siam, device, gallery_dict, probe_dict_low)

    # N vs high
    probe_dict_high =  metrics.filter_descriptordict_scans(probe_dict, bu3dfe_filter_high_intensity)
    y_true_high, y_score_high  = metrics.generate_metric_siamese_roc_bal(siam, device, gallery_dict, probe_dict_high)

    # N vs ALL
    y_true_all, y_score_all  = metrics.generate_metric_siamese_roc_bal(siam, device, gallery_dict, probe_dict)

    # All vs All
    y_true_all_v_all, y_score_all_v_all = metrics.generate_metric_siamese_roc_bal_all(siam, device, descriptor_dict)

    y_trues = [y_true_low, y_true_high, y_true_all, y_true_all_v_all]
    y_scores = [y_score_low, y_score_high, y_score_all, y_score_all_v_all]
    labels = ["Neutral vs. Low-Intensity", "Neutral vs. High-Intensity", "Neutral vs. All", "All vs. All"]

    verification_rates, accuracies, fprs = generate_verification_rates(y_trues, y_scores)
    ap = sklearn.metrics.average_precision_score(y_trues[2].cpu(), y_scores[2].cpu())
    fprs, recalls, roc_aucs = generate_roc_aucs(y_trues, y_scores)

    return generate_roc_from_lists(fprs, recalls, roc_aucs, labels), \
           generate_roc_from_lists(fprs, recalls, roc_aucs, labels, log=True), \
           verification_rates[2], roc_aucs[2], generate_acc_fpr(fprs, accuracies, labels), ap

def bosphorus_generate_roc(descriptor_dict, siam, device):
    gallery_dict, probe_dict = metrics.split_gallery_set_vs_probe_set_bosphorus(descriptor_dict)

    # Assert gallery is correctly layed out
    for ident, ident_dict in gallery_dict.items():
        assert len(ident_dict) == 1

    # N vs low
    probe_dict_low =  metrics.filter_descriptordict_scans(probe_dict, bosphorus_filter_neutral)
    y_true_low, y_score_low  = metrics.generate_metric_siamese_roc_bal(siam, device, gallery_dict, probe_dict_low)

    # N vs high
    probe_dict_high =  metrics.filter_descriptordict_scans(probe_dict, bosphorus_filter_non_neutral)
    y_true_high, y_score_high  = metrics.generate_metric_siamese_roc_bal(siam, device, gallery_dict, probe_dict_high)

    # N vs ALL
    y_true_all, y_score_all  = metrics.generate_metric_siamese_roc_bal(siam, device, gallery_dict, probe_dict)

    # All vs all
    y_true_all_v_all, y_score_all_v_all = metrics.generate_metric_siamese_roc_bal_all(siam, device, descriptor_dict)


    y_trues = [y_true_low, y_true_high, y_true_all, y_true_all_v_all]
    y_scores = [y_score_low, y_score_high, y_score_all, y_score_all_v_all]
    labels = ["Neutral vs. Neutral", "Neutral vs. Non-neutral", "Neutral vs. All", "All vs. All"]
    
    verification_rates, accuracies, fprs = generate_verification_rates(y_trues, y_scores)
    ap = sklearn.metrics.average_precision_score(y_trues[2].cpu(), y_scores[2].cpu())
    fprs, recalls, roc_aucs = generate_roc_aucs(y_trues, y_scores)

    return generate_roc_from_lists(fprs, recalls, roc_aucs, labels), \
           generate_roc_from_lists(fprs, recalls, roc_aucs, labels, log=True), \
           verification_rates[2], roc_aucs[2], generate_acc_fpr(fprs, accuracies, labels), ap

def frgc_generate_roc(descriptor_dict, siam, device):
    gallery_dict, probe_dict = metrics.split_gallery_set_vs_probe_set_frgc(descriptor_dict)

    # Assert gallery is correctly layed out
    for ident, ident_dict in gallery_dict.items():
        assert len(ident_dict) == 1

    # Cannot filter probe as there is no markings on what type it is 
    # offloading may help if it runs out of memory here 
    y_true_all, y_score_all  = metrics.generate_metric_siamese_roc_bal(siam, device, gallery_dict, probe_dict, offload=False)
    y_true_all_v_all, y_score_all_v_all = metrics.generate_metric_siamese_roc_bal_all(siam, device, descriptor_dict)


    # verification_rates, accuracies, fpr = generate_veritication_rate(y_true_all, y_score_all)
    

    y_trues = [y_true_all, y_true_all_v_all]
    y_scores = [y_score_all, y_score_all_v_all]
    labels = ["First vs. Rest", "All vs. All"]

    verification_rates, accuracies, fprs = generate_verification_rates(y_trues, y_scores)
    ap = sklearn.metrics.average_precision_score(y_trues[0].cpu(), y_scores[0].cpu())
    fprs, recalls, roc_aucs = generate_roc_aucs(y_trues, y_scores)

    return generate_roc_from_lists(fprs, recalls, roc_aucs, labels), \
           generate_roc_from_lists(fprs, recalls, roc_aucs, labels, log=True), \
           verification_rates[0], roc_aucs[0], generate_acc_fpr(fprs, accuracies, labels), ap


def face3d_generate_roc(descriptor_dict, siam, device):  # 3dface
    gallery_dict, probe_dict = metrics.split_gallery_set_vs_probe_set_3dface(descriptor_dict)

    # Assert gallery is correctly layed out
    for ident, ident_dict in gallery_dict.items():
        assert len(ident_dict) == 1

    y_true_all, y_score_all  = metrics.generate_metric_siamese_roc_bal(siam, device, gallery_dict, probe_dict)
    y_true_all_v_all, y_score_all_v_all = metrics.generate_metric_siamese_roc_bal_all(siam, device, descriptor_dict)

    y_trues = [y_true_all, y_true_all_v_all]
    y_scores = [y_score_all, y_score_all_v_all]
    labels = ["First vs. Second", "All vs. All"]

    verification_rates, accuracies, fprs = generate_verification_rates(y_trues, y_scores)
    ap = sklearn.metrics.average_precision_score(y_trues[0].cpu(), y_scores[0].cpu())
    fprs, recalls, roc_aucs = generate_roc_aucs(y_trues, y_scores)

    return generate_roc_from_lists(fprs, recalls, roc_aucs, labels), \
           generate_roc_from_lists(fprs, recalls, roc_aucs, labels, log=True), \
           verification_rates[0], roc_aucs[0], generate_acc_fpr(fprs, accuracies, labels), ap


def face3d_generate_cmc(descriptor_dict, siam, device):
    gallery_dict, probe_dict = metrics.split_gallery_set_vs_probe_set_3dface(descriptor_dict)

    # Assert gallery is correctly layed out
    for ident, ident_dict in gallery_dict.items():
        assert len(ident_dict) == 1


    rank_tp_dict  = metrics.generate_metirc_siamese_rank1_cmc(siam, device, gallery_dict, probe_dict)
    ir_by_rank_all = metrics.generate_identification_rate_by_rank(rank_tp_dict)

    ir_ranks = [ir_by_rank_all]
    labels = ["First vs. Second"]
    return generate_cmc_from_lists(ir_ranks, labels)



def all_v_all_generate_roc(descriptor_dict, siam, device):
    y_true, y_score = metrics.generate_metric_siamese_roc_bal_all(siam, device, descriptor_dict)

    verification_rates, accuracies, fpr = generate_veritication_rate(y_true, y_score)
    ap = sklearn.metrics.average_precision_score(y_true.cpu(), y_score.cpu())
    fprs, recalls, roc_aucs = generate_roc_aucs([y_true], [y_score])

    labels = ["All vs. All"]
    return None, None, verification_rates, roc_aucs[0], None, ap
