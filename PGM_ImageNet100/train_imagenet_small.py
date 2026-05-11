import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

import os
import random
import numpy as np

# -------------------------
# Determinism (important)
# -------------------------
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_result(seed, lr, epochs, acc, log_file="results.csv"):
    header = "seed,lr,epochs,acc\n"

    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write(header)

    with open(log_file, "a") as f:
        f.write(f"{seed},{lr},{epochs},{acc:.6f}\n")


# -------------------------
# Evaluation (NEW)
# -------------------------
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            out = model(x)
            pred = out.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total


# -------------------------
# Main training function
# -------------------------
def train_one(seed, train_dir, val_dir, epochs=5, lr=5e-4):
    set_seed(seed)

    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    ])

    # -------- datasets --------
    train_set = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    val_set   = torchvision.datasets.ImageFolder(val_dir, transform=transform)

    # -------- loaders --------
    train_loader = DataLoader(
        train_set,
        batch_size=64,
        shuffle=True,   # IMPORTANT: keep deterministic for pairing
        num_workers=4
    )

    val_loader = DataLoader(
        val_set,
        batch_size=64,
        shuffle=False,
        num_workers=4
    )

    # -------- model --------
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False

    # replace head
    model.fc = nn.Linear(model.fc.in_features, len(train_set.classes))
    nn.init.normal_(model.fc.weight, 0, 0.01)
    nn.init.zeros_(model.fc.bias)
    
    # train only head
    for param in model.fc.parameters():
        param.requires_grad = True
    
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr, weight_decay=1e-4)

    # -------- training --------
    model.train()
    for epoch in range(epochs):
        correct, total = 0, 0

        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_acc = correct / total
        print(f"[seed {seed}] epoch {epoch}: train_acc={train_acc:.4f}")

    # -------- validation --------
    val_acc = evaluate(model, val_loader)
    print(f"[seed {seed}] FINAL_VAL_ACC={val_acc:.4f}")

    return val_acc


# -------------------------
# Entry
# -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)

    args = parser.parse_args()

    acc = train_one(
        args.seed,
        args.train_dir,
        args.val_dir,
        args.epochs,
        args.lr
    )

    print(f"FINAL_ACC {acc}")

    log_result(args.seed, args.lr, args.epochs, acc)