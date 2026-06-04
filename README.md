# Computer Vision Competition Starter Framework

Welcome to the Computer Vision Competition! This framework provides a baseline to help you get started with training your own models and generating predictions.

## Project Structure

- `data/`: Place your training and testing images here.
  - `train/`: Training images.
  - `test/`: Test images (for submission).
  - `labels_train.csv`: Training labels in `image_id,label` format.
- `models/`: Contains the model architecture.
  - `baseline_model.py`: A ResNet-18 baseline model.
- `dataset.py`: PyTorch Dataset class for loading data.
- `train.py`: Script to train your model.
- `predict.py`: Script to generate `submission.csv`.
- `utils.py`: Helper functions for accuracy and transformations.
- `requirements.txt`: List of required Python packages.

## Getting Started

### 1. Virtual Environment Setup (Recommended)
It is highly recommended to use a virtual environment to manage dependencies.

#### Windows:
```powershell
python -m venv venv
.\venv\Scripts\activate
```

#### macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Setup
Install the required dependencies within your virtual environment:
```bash
pip install -r requirements.txt
```

### 3. Prepare Data
If your source images are already organized as one folder per class, use
`prepare_data.py` to create the competition layout automatically:

```bash
python prepare_data.py path/to/folder_per_class_dataset --output-dir data --overwrite
```

Example input:

```text
raw_dataset/
  Cats/
    cat_001.jpg
    cat_002.jpg
  Dogs/
    dog_001.jpg
    dog_002.jpg
```

The script writes:

- `data/train/`: copied training images
- `data/test/`: copied test images
- `data/labels_train.csv`: `image_id,label` training labels
- `data/secret_ground_truth.csv`: held-out test labels for organizers
- `data/label_map.json`: integer-label to class-name mapping

By default, `20%` of each class is held out for the test split. Adjust it with:

```bash
python prepare_data.py path/to/folder_per_class_dataset --test-ratio 0.3 --seed 123 --output-dir data --overwrite
```

If your data is already in competition format, ensure your images are in the
`data/train` and `data/test` folders, and `labels_train.csv` is in the `data/`
directory.

### 4. Training
To train the baseline model, run:
```bash
python train.py
```
This will:
- Load the training data.
- Split it into training and validation sets.
- Train the model for 5 epochs.
- Save the best model as `model.pth`.

### 5. Prediction
Once the model is trained, generate your submission file:
```bash
python predict.py
```
This will:
- Load `model.pth`.
- Run inference on the images in `data/test`.
- Save the results to `submission.csv`.

## Submission
Upload the generated `submission.csv` to the web-based evaluation system.

## Tips for Improving Results
1. **Change the Model**: Try different architectures in `models/baseline_model.py`.
2. **Data Augmentation**: Add more transforms in `utils.py` (e.g., RandomHorizontalFlip, ColorJitter).
3. **Hyperparameter Tuning**: Experiment with learning rates, batch sizes, and number of epochs in `train.py`.
4. **Transfer Learning**: Use pretrained weights to speed up convergence and improve accuracy.

Good luck!
