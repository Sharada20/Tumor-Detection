import os
import argparse
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = (224, 224)
NUM_CLASSES = 4

def get_dataloaders(data_dir, batch_size, val_split=0.15):
    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2), # optional medical color jittering
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])
    full_train_dataset = datasets.ImageFolder(os.path.join(data_dir, "Training"), transform=train_transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "Testing"), transform=test_transform)
    
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader, full_train_dataset

def build_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(512, NUM_CLASSES)
    )
    return model

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True
    return model

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"run_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    
    train_loader, val_loader, test_loader, full_data = get_dataloaders(args.data_dir, args.batch)
    
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
    
    warmup_epochs = min(args.epochs, max(1, args.epochs // 3))
    
    for phase_name, epochs, unfreeze in [("Phase 1 (Head)", warmup_epochs, False), ("Phase 2 (Full)", args.epochs - warmup_epochs, True)]:
        if epochs <= 0: continue
        print(f"\n--- {phase_name} ({epochs} epochs) ---")
        if unfreeze:
            model = unfreeze_model(model)
            optimizer = optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == lbls)
                
            train_acc = correct.double() / len(train_loader.dataset)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {running_loss/len(train_loader.dataset):.4f} - Acc: {train_acc:.4f}")

    print("\nEvaluating on Test Set...")
    model.eval()
    all_preds, all_lbls, all_probs = [], [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_lbls.extend(lbls.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    print(classification_report(all_lbls, all_preds, target_names=CLASS_NAMES))
    
    torch.save(model.state_dict(), os.path.join(out_dir, "brain_tumor_model.pth"))
    torch.save(model.state_dict(), "brain_tumor_model.pth")
    print(f"\nModel saved locally as brain_tumor_model.pth!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data_preproc")
    parser.add_argument("--output_dir", type=str, default="runs")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=16) 
    args = parser.parse_args()  
    train(args)
