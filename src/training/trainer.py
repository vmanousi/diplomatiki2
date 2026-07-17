from pathlib import Path

import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        checkpoint_dir,
        tensorboard_dir=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(str(tensorboard_dir)) if tensorboard_dir else None

        self.best_val_f1 = 0.0

    def train_one_epoch(self):
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(self.train_loader, desc="Training"):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * labels.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def validate(self):
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * labels.size(0)

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        avg_loss = total_loss / total
        accuracy = correct / total

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )

        return avg_loss, accuracy, precision, recall, f1

    def save_checkpoint(self, epoch, val_acc, val_f1):
        checkpoint_path = self.checkpoint_dir / "best_model.pt"

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_acc": val_acc,
                "val_f1": val_f1,
            },
            checkpoint_path,
        )

    def fit(self, epochs, early_stopping_patience=None):
        """
        Best checkpoint and early stopping are both driven by macro-F1
        (not accuracy) — with per-class imbalance, accuracy can reward a
        checkpoint that ignores rare classes as long as it nails the
        common ones, which is the opposite of what a "detect pathological
        findings across N classes" task should optimize for.

        early_stopping_patience:
            If set, stop training after this many consecutive epochs
            without a val_f1 improvement. None (default) trains the full
            number of epochs, matching the previous behaviour.
        """

        history = []
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")

            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc, val_precision, val_recall, val_f1 = self.validate()

            print(
                f"Train loss: {train_loss:.4f} | "
                f"Train acc: {train_acc:.4f} | "
                f"Val loss: {val_loss:.4f} | "
                f"Val acc: {val_acc:.4f} | "
                f"Val F1 (macro): {val_f1:.4f}"
            )

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                }
            )

            if self.writer:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Accuracy/train", train_acc, epoch)
                self.writer.add_scalar("Accuracy/val", val_acc, epoch)
                self.writer.add_scalar("Precision/val_macro", val_precision, epoch)
                self.writer.add_scalar("Recall/val_macro", val_recall, epoch)
                self.writer.add_scalar("F1/val_macro", val_f1, epoch)

            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.save_checkpoint(epoch, val_acc, val_f1)
                print("Saved new best model.")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (
                early_stopping_patience is not None
                and epochs_without_improvement >= early_stopping_patience
            ):
                print(
                    "Early stopping: no val_f1 improvement for "
                    f"{early_stopping_patience} epochs."
                )
                break

        if self.writer:
            self.writer.close()

        return history