from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def evaluate_model(model, dataloader, device, idx_to_label, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]

    acc = accuracy_score(all_labels, all_preds)

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_dir / "classification_report.csv")

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(12, 12))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )
    disp.plot(
        xticks_rotation=90,
        cmap="Blues",
        values_format="d",
    )
    plt.title(f"Confusion Matrix - Accuracy: {acc:.4f}")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    print(f"Test accuracy: {acc:.4f}")
    print(report_df)

    return acc, report_df, cm