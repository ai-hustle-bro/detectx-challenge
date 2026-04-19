import torch
import pandas as pd
from torch.utils.data import DataLoader
from dataset import CompetitionDataset
from models.baseline_model import get_model
import os
from tqdm import tqdm

# Configuration
TEST_DIR = 'data/test'
MODEL_PATH = 'model.pth'
SUBMISSION_FILE = 'submission.csv'
BATCH_SIZE = 16
NUM_CLASSES = 10 # Should match the training config

def predict():
    # 1. Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Please train the model first.")
        return

    # 2. Load Test Dataset
    test_dataset = CompetitionDataset(data_dir=TEST_DIR, is_test=True)
    if len(test_dataset) == 0:
        print(f"Warning: No images found in {TEST_DIR}.")
        return
        
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    predictions = []
    image_ids = []

    print("Running inference...")
    test_bar = tqdm(test_loader, desc="Predicting")

    # 4. Inference Loop
    with torch.no_grad():
        for images, ids in test_bar:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            predictions.extend(preds.cpu().numpy())
            image_ids.extend(ids)

    # 5. Save to CSV
    submission_df = pd.DataFrame({
        'image_id': image_ids,
        'label': predictions
    })
    
    submission_df.to_csv(SUBMISSION_FILE, index=False)
    print(f"Predictions saved to {SUBMISSION_FILE}")

if __name__ == "__main__":
    predict()
