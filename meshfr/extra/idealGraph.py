import matplotlib.pyplot as plt
import numpy as np

#  https://matplotlib.org/stable/gallery/color/named_colors.html

create_cmc = False 
create_roc = False 
create_sigmoid = False
create_relu = True


if create_cmc:
    # CMC
    array = [1.0]*9
    array[0:4] = [0.98, 0.995, 0.998, 0.9995]

    # fig, ax = plt.subplots()
    plt.title("CMC Curve")
    plt.plot(list(range(1, len(array)+1)), array, marker="o", lw=2, color="darkorange", label = "Almost ideal", clip_on=False)
    plt.ylabel("Identification Rate")
    plt.xlabel("Rank")
    # plt.legend(loc="lower right")
    plt.xlim(left=1, right=len(array))
    plt.ylim(top=1.0005)
    plt.grid(True)
    plt.locator_params(axis="x", integer=True)
    plt.savefig("ideal-cmc.pdf", bbox_inches='tight')
    plt.close()

if create_roc:
    # ROC
    import random 
    from sklearn.metrics import roc_curve, auc
    random.seed(0)
    y_true = [] 
    y_score = []
    def add(tf, score_t, amount, pm):
        for i in range(amount):
            score = random.uniform(score_t-pm, score_t+pm)
            y_true.append(tf)
            y_score.append(score)

    add(1, 0.8, 80, 0.1)  # True 0.7-0.9
    add(1, 0.65, 20, 0.05) # True 0.6-0.7
    add(0, 0.4, 200, 0.23)
    # add(1, 0.4, 1, 0.01)

    false_positive_rate, recall, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(false_positive_rate, recall)
    z = 1e-10
    plt.title("FAR - VR")
    plt.plot(false_positive_rate, recall, color="darkorange", lw=2, label = "AUC = %0.3f" % roc_auc)  # b
    plt.legend(loc="lower right")
    #plt.plot([0+z,1-z], [0,1], lw=2, linestyle="--")
    x = np.linspace(0,1,101)
    plt.plot(x, x, lw=2, linestyle="--")  # RNG line which works for log too
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.ylabel("True Positive Rate (Recall)")
    plt.xlabel("False Acceptance Rate")
    plt.grid(True, which="both") #  ls="-"
    plt.savefig("ideal-roc.pdf", bbox_inches='tight')

    plt.xscale("log", nonpositive="clip")
    plt.xlabel("False Acceptance Rate (Log scale)")
    plt.xlim(left=1e-3, right=1.05)
    plt.ylim([0.9,1.001])
    plt.savefig("ideal-roc-log.pdf", bbox_inches='tight')
    plt.close()


if create_sigmoid:
    # Sigmoid 
    x = np.linspace(-12, 12, 201)
    z = 1/(1 + np.exp(-x))
    
    plt.plot(x, z)
    plt.xlabel("x")
    plt.ylabel("Sigmoid(x)")
    plt.grid(True, which="both")
    plt.xlim(left=-11, right=11)
    plt.savefig("sigmoid.pdf", bbox_inches='tight')

if create_relu:
    x = np.linspace(-12, 12, 201)
    z = np.maximum(0, x)
    plt.plot(x, z)
    plt.xlabel("x")
    plt.ylabel("ReLU(x)")
    plt.grid(True, which="both")
    plt.xlim(left=-10, right=10)
    plt.ylim(bottom=-0.4, top=10)
    plt.savefig("relu.pdf", bbox_inches='tight')