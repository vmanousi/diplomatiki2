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


def evaluate_model(model, dataloader, device, output_dir, class_names=None):
    """
    Evaluate a trained model and save all evaluation artifacts.

    class_names:
        Optional list of class name strings, ordered by class index
        (e.g. dataset.idx_to_label built into a list). When given, all
        saved artifacts are labeled with real class names instead of
        raw integer indices. When None (e.g. GastroVision, which has no
        string label available), falls back to numeric indices exactly
        as before.

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

    # Explicit labels (and target_names, when available) so every class
    # gets a row/column even if it happens to have zero true or predicted
    # samples in this particular evaluation, and so results are labeled
    # with real class names instead of raw integer indices.
    label_indices = (
        list(range(len(class_names)))
        if class_names is not None
        else None
    )

    report = classification_report(
        all_labels,
        all_preds,
        labels=label_indices,
        target_names=class_names,
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
        labels=label_indices,
    )

    pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names,
    ).to_csv(
        output_dir / "confusion_matrix.csv",
        index=class_names is not None,
    )

    plt.figure(figsize=(10, 8))

    plt.imshow(
        cm,
        interpolation="nearest",
    )

    plt.title("Confusion Matrix")

    plt.colorbar()

    if class_names is not None:
        tick_positions = range(len(class_names))
        plt.xticks(tick_positions, class_names, rotation=90)
        plt.yticks(tick_positions, class_names)

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
