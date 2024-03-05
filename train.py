import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from math import e
from tqdm import tqdm
from torchmetrics import Accuracy, Precision, Recall, F1Score, MetricCollection
from models import *
from data_pipeline import get_dataloaders
from sklearn.metrics.pairwise import haversine_distances as hsine


def train(model, loader, optimizer, centroids, metrics_group, tau=50, device="cpu"):

    epoch_loss = 0.0
    model.train()
    for inputs, labels, coords in tqdm(loader):
        # inputs = {k:v.to(device) for k, v in inputs.items()}
        inputs = inputs.to(device)
        out = F.softmax(model(inputs), dim=-1)
        logits = -torch.log(out+1e-10) # B x 42

        input_coords = coords.detach().numpy()
        a = 6371 * hsine(np.deg2rad(input_coords), np.deg2rad(centroids)) # B x 42
        b = np.diag(6371 * hsine(np.deg2rad(centroids[labels.detach().numpy()]), np.deg2rad(coords)))[:, np.newaxis] # B x 1
        weights = torch.tensor(e**(-(a - b) / tau)).detach().to(device) # B x 42


        loss = torch.mean(torch.sum(logits * weights, dim=1))
        loss.backward()
        optimizer.step()

        labels = labels.to(device)
        metrics_group.update(out, labels)
        epoch_loss += loss.item()

    return epoch_loss / len(loader), metrics_group.compute()

def evaluate(model, loader, centroids, metrics_group, tau=50, device="cpu"):

    epoch_loss = 0.0
    model.eval()

    with torch.no_grad():
        for inputs, labels, coords in tqdm(loader):
            # inputs = {k:v.to(device) for k, v in inputs.items()}
            inputs = inputs.to(device)

            out = F.softmax(model(inputs), dim=-1)
            logits = -torch.log(out+1e-10) # B x 42

            input_coords = coords.detach().numpy()
            a = 6371 * hsine(np.deg2rad(input_coords), np.deg2rad(centroids)) # B x 42
            b = np.diag(6371 * hsine(np.deg2rad(centroids[labels.detach().numpy()]), np.deg2rad(coords)))[:, np.newaxis] # B x 1
            weights = torch.tensor(e**(-(a - b) / tau)).detach().to(device) # B x 42

            loss = torch.mean(torch.sum(logits * weights, dim=1))

            labels = labels.to(device)
            metrics_group.update(out, labels)
            epoch_loss += loss.item()

    return epoch_loss / len(loader), metrics_group.compute()


def inference(model, img_name, transform, class_names, device="cpu"):

    inputs = transform(
        torch.tensor(cv2.imread(img_name)).permute(2,0,1)
    ).unsqueeze(0).to(device)

    model.eval()
    model = model.to(device)
    with torch.no_grad():
        outputs = F.softmax(model(inputs), dim=-1)[0]

    top_preds = torch.topk(outputs, k=5)
    probs = top_preds.values.tolist()
    indices = top_preds.indices.tolist()
    classes = [class_names[idx] for idx in indices]
    print([(a, b) for a,b in zip(classes, probs)])

def main():

    db_root = "geo_dbv2"
    (train_loader, val_loader, test_loader), centroids, num_classes = get_dataloaders(db_root, batch_size=32)
    class_names = list(sorted(os.listdir(db_root)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GeoFinderV1(num_classes=num_classes)
    model = model.to(device)

    metrics_group = MetricCollection(
        [
            Accuracy(task="multiclass",num_classes=num_classes),
            Precision(task="multiclass",num_classes=num_classes),
            Recall(task="multiclass",num_classes=num_classes),
            F1Score(task="multiclass",num_classes=num_classes)
        ]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    epochs = 30
    best_score = None
    best_state_dict = None

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        train_loss, train_metrics = train(
            model,
            train_loader,
            optimizer,
            centroids,
            metrics_group,
            tau=100,
            device=device
        )
        metrics_group.reset()
        
        valid_loss, valid_metrics = evaluate(
            model,
            val_loader,
            centroids,
            metrics_group,
            tau=100,
            device=device
        )
        metrics_group.reset()

        valid_f1 = valid_metrics["MulticlassF1Score"].cpu().item()
        if (best_score is None) or (valid_f1 > best_score):
            best_score = valid_f1
            best_state_dict = model.state_dict()

        print("Train")
        print(f"Loss: {train_loss:.3f}", end=", ")
        print(", ".join([f"{k}: {v.cpu().item() * 100:.2f}" for k, v in train_metrics.items()]))

        print("Validation")
        print(f"Loss: {valid_loss:.3f}", end=", ")
        print(", ".join([f"{k}: {v.cpu().item() * 100:.2f}" for k, v in valid_metrics.items()]))
        print()

    model.load_state_dict(best_state_dict)
    test_loss, test_metrics = evaluate(
        model,
        test_loader,
        centroids,
        metrics_group,
        tau=50,
        device=device
    )
    metrics_group.reset()

    print("Test")
    print(f"Loss: {test_loss:.3f}", end=", ")
    print(", ".join([f"{k}: {v.cpu().item() * 100:.2f}" for k, v in test_metrics.items()]))




if __name__=="__main__":
    main()