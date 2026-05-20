import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

from models.hfs_net import HFSNet


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SyntheticTrafficDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples=2000, seq_len=128, stats_dim=16, num_classes=4):
        self.seq = np.random.randn(n_samples, seq_len, 2).astype(np.float32)
        self.stats = np.random.randn(n_samples, stats_dim).astype(np.float32)

        # Create label signal from both branches to simulate fusion gains.
        seq_signal = self.seq[:, :, 0].mean(axis=1) + 0.5 * self.seq[:, :, 1].std(axis=1)
        stats_signal = self.stats[:, :4].sum(axis=1)
        combined = seq_signal + 0.8 * stats_signal

        bins = np.quantile(combined, np.linspace(0, 1, num_classes + 1))
        y = np.digitize(combined, bins[1:-1], right=False)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.seq[idx], self.stats[idx], self.y[idx]


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for seq, stats, y in loader:
            seq = seq.to(device)
            stats = stats.to(device)
            y = y.to(device)
            logits, _ = model(seq, stats)
            pred = torch.argmax(logits, dim=-1)
            preds.extend(pred.cpu().numpy().tolist())
            labels.extend(y.cpu().numpy().tolist())

    return {
        "acc": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--stats_dim", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = SyntheticTrafficDataset(4000, args.seq_len, args.stats_dim, args.num_classes)
    val_set = SyntheticTrafficDataset(1000, args.seq_len, args.stats_dim, args.num_classes)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size)

    model = HFSNet(stats_dim=args.stats_dim, num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for seq, stats, y in train_loader:
            seq = seq.to(device)
            stats = stats.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits, _ = model(seq, stats)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        metrics = evaluate(model, val_loader, device)
        avg_loss = running_loss / max(1, len(train_loader))
        print(
            f"epoch={epoch} loss={avg_loss:.4f} val_acc={metrics['acc']:.4f} val_f1={metrics['f1_macro']:.4f}"
        )

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best.pt"))

    print(f"best_val_f1={best_f1:.4f}")


if __name__ == "__main__":
    main()
