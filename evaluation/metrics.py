# evaluation/metrics.py
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

def compute_f1_score(true, pred):
    return f1_score(true, pred, average="macro")

def compute_auc(true, pred):
    try:
        return roc_auc_score(true, pred, multi_class="ovr")
    except:
        return 0.0  # fallback for binary or uniform labels

def compute_accuracy(true, pred):
    return accuracy_score(true, pred)
