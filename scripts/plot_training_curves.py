from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


experiment_dir = Path("experiments/exp01_resnet18")
history_path = experiment_dir / "history.csv"
plots_dir = experiment_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

history = pd.read_csv(history_path)

# Loss curve
plt.figure(figsize=(8, 5))
plt.plot(history["epoch"], history["train_loss"], marker="o", label="Train loss")
plt.plot(history["epoch"], history["val_loss"], marker="o", label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ResNet18 Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plots_dir / "loss_curve.png", dpi=300)
plt.close()

# Accuracy curve
plt.figure(figsize=(8, 5))
plt.plot(history["epoch"], history["train_acc"], marker="o", label="Train accuracy")
plt.plot(history["epoch"], history["val_acc"], marker="o", label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ResNet18 Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plots_dir / "accuracy_curve.png", dpi=300)
plt.close()

print(f"Saved plots to: {plots_dir}")