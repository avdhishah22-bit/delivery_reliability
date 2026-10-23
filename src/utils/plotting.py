"""Plot helpers."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_roc_pr(y_true, probs: dict, path):
    """probs: {model_name: predicted probabilities on the test set}."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        pr, rc, _ = precision_recall_curve(y_true, p)
        ax[0].plot(fpr, tpr, label=name)
        ax[1].plot(rc, pr, label=name)
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.5)
    ax[0].set(title="ROC (test)", xlabel="FPR", ylabel="TPR")
    ax[1].set(title="Precision-Recall (test)", xlabel="Recall", ylabel="Precision")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
