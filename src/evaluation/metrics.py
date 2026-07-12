from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm


def evaluate_model(model, dataloader, device, output_dir):
    """
    Evaluate a trained model and save all evaluation artifacts.

    Saves:
        - classification_report.csv
        - confusion_matrix.csv
        - confusion_matrix.png
        - predictions.csv
        - metrics.json
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():

        for images, labels in tqdm(dataloader, desc="Final evaluation"):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.max(dim=1).values.cpu().numpy())

    ############################################################
    # Metrics
    ############################################################

    accuracy = accuracy_score(all_labels, all_preds)

    report = classification_report(
        all_labels,
        all_preds,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(
        output_dir / "classification_report.csv"
    )

    ############################################################
    # Confusion Matrix
    ############################################################

    cm = confusion_matrix(
        all_labels,
        all_preds,
    )

    pd.DataFrame(cm).to_csv(
        output_dir / "confusion_matrix.csv",
        index=False,
    )

    plt.figure(figsize=(10, 8))

    plt.imshow(
        cm,
        interpolation="nearest",
    )

    plt.title("Confusion Matrix")

    plt.colorbar()

    plt.xlabel("Predicted class")
    plt.ylabel("True class")

    plt.tight_layout()

    plt.savefig(
        output_dir / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    ############################################################
    # Predictions
    ############################################################

    predictions = pd.DataFrame(
        {
            "true_label": all_labels,
            "predicted_label": all_preds,
            "confidence": all_probs,
        }
    )

    predictions.to_csv(
        output_dir / "predictions.csv",
        index=False,
    )

    ############################################################
    # Summary Metrics
    ############################################################

    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall": float(report["weighted avg"]["recall"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics
