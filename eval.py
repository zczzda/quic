import argparse

import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

from models.hfs_net import HFSNet
from train import SyntheticTrafficDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--stats_dim", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_set = SyntheticTrafficDataset(1000, args.seq_len, args.stats_dim, args.num_classes)
    loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size)

    model = HFSNet(stats_dim=args.stats_dim, num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for seq, stats, y in loader:
            seq = seq.to(device)
            stats = stats.to(device)
            logits, _ = model(seq, stats)
            pred = torch.argmax(logits, dim=-1)
            preds.extend(pred.cpu().numpy().tolist())
            labels.extend(y.numpy().tolist())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    print(f"test_acc={acc:.4f} test_f1_macro={f1:.4f}")
    print(classification_report(labels, preds, digits=4))


if __name__ == "__main__":
    main()
