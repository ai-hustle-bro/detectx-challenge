import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import CompetitionDataset
from models.baseline_model import get_model
from utils import get_accuracy
import os
from tqdm import tqdm

# Configuration
DATA_DIR = 'data/train'
CSV_FILE = 'data/labels_train.csv'
MODEL_SAVE_PATH = 'model.pth'
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2

def train():
    # 1. Load Dataset
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found. Please ensure your data is in the correct folder.")
        return

    full_dataset = CompetitionDataset(data_dir=DATA_DIR, csv_file=CSV_FILE)
    
    # 2. Split into Train and Validation
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Initialize Model, Loss, and Optimizer
    num_classes = len(full_dataset.labels_df['label'].unique())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = get_model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting training on {device}...")

    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += get_accuracy(outputs, labels)
            train_bar.set_postfix(loss=running_loss/len(train_loader), acc=running_acc/len(train_loader))

        # Validation
        model.eval()
        val_acc = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_acc += get_accuracy(outputs, labels)
                val_bar.set_postfix(val_acc=val_acc/len(val_loader))

        print(f"Epoch [{epoch+1}/{EPOCHS}] - "
              f"Loss: {running_loss/len(train_loader):.4f}, "
              f"Train Acc: {running_acc/len(train_loader):.4f}, "
              f"Val Acc: {val_acc/len(val_loader):.4f}")

    # 5. Save Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
