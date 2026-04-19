import torch.nn as nn
from torchvision import models

def get_model(num_classes=10, pretrained=True):
    """
    Returns a ResNet18 model. 
    ResNet18 is lightweight and suitable for CPU training in limited time.
    """
    # Load a pretrained ResNet18 or a fresh one
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=None)

    # Modify the final fully connected layer to match the number of classes in the competition
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

if __name__ == "__main__":
    # Test model creation
    model = get_model(num_classes=2)
    print(model)
