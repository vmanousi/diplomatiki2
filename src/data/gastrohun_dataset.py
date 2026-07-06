from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class GastroHUNDataset(Dataset):
    def __init__(self, images_root, csv_path, split="Train",
                 label_column="Complete agreement", transform=None):
        self.images_root = Path(images_root)
        self.csv_path = Path(csv_path)
        self.split = split
        self.label_column = label_column
        self.transform = transform

        df = pd.read_csv(self.csv_path)
        df.columns = df.columns.str.strip()
        df = df[df["set_type"] == split].copy()
        df = df.dropna(subset=[label_column])
        df = df[df[label_column].astype(str).str.strip() != ""]

        self.labels = sorted(df[label_column].unique())
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}
        self.idx_to_label = {i: label for label, i in self.label_to_idx.items()}

        df["label_idx"] = df[label_column].map(self.label_to_idx)
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        patient_id = str(int(row["num patient"]))
        filename = row["filename"]
        label = int(row["label_idx"])

        image_path = self.images_root / patient_id / filename

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label