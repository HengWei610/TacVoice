# evaluation/evaluate_asr.py
from evaluation.metrics import compute_f1_score, compute_auc, compute_accuracy
from utils.dataloader import load_test_data
import torch

def evaluate_model(model, test_loader):
    all_preds, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["audio_features"]
            labels = batch["labels"]

            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)

            all_preds.extend(predictions.tolist())
            all_labels.extend(labels.tolist())

    f1 = compute_f1_score(all_labels, all_preds)
    auc = compute_auc(all_labels, all_preds)
    acc = compute_accuracy(all_labels, all_preds)

    print(f"F1 Score: {f1:.4f} | AUC: {auc:.4f} | Accuracy: {acc:.4f}")
