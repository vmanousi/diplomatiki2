from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_curves(history_csv, output_dir):
    history_csv = Path(history_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = pd.read_csv(history_csv)
    epochs = history["epoch"]

    # Loss curve
    plt.figure()
    plt.plot(epochs, history["train_loss"], marker="o", label="Train loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=300)
    plt.close()

    # Accuracy curve
    plt.figure()
    plt.plot(epochs, history["train_acc"], marker="o", label="Train accuracy")
    plt.plot(epochs, history["val_acc"], marker="o", label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=300)
    plt.close()

    # Precision / recall / F1 curve (validation, macro-averaged)
    plt.figure()
    plt.plot(epochs, history["val_precision"], marker="o", label="Val precision (macro)")
    plt.plot(epochs, history["val_recall"], marker="o", label="Val recall (macro)")
    plt.plot(epochs, history["val_f1"], marker="o", label="Val F1 (macro)")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation precision / recall / F1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "val_prf_curve.png", dpi=300)
    plt.close()
