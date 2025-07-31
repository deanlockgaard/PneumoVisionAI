# src/models.py
"""
PneumoVisionAI - Model Architectures
Defines the CNN models used for classification.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

# --- 1. Baseline CNN Model ---
class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for baseline performance.
    """
    def __init__(self, num_classes: int = 2):
        super(SimpleCNN, self).__init__()
        
        self.conv_layer = nn.Sequential(
            # Conv block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv block 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # After 3 max-pooling layers of stride 2, an image of 224x224 becomes 28x28
        # 224 -> 112 -> 56 -> 28
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x

# --- 2. Pre-trained Model Loader ---
def get_pretrained_model(
    model_name: str = "resnet50", 
    num_classes: int = 2, 
    pretrained: bool = True,
    freeze_layers: bool = True
) -> nn.Module:
    """
    Loads a pre-trained model and adapts its final layer for our classification task.

    Args:
        model_name (str): The name of the model to load (e.g., 'resnet50', 'efficientnet_b0').
        num_classes (int): The number of output classes.
        pretrained (bool): Whether to use pre-trained weights from ImageNet.
        freeze_layers (bool): Whether to freeze the weights of the convolutional base.
        
    Returns:
        nn.Module: The adapted pre-trained model.
    """
    # Load the pre-trained model
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    else:
        raise ValueError(f"Model '{model_name}' not supported.")
        
    # Freeze the convolutional layers if specified
    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace the final classification layer
    if model_name == "resnet50":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "efficientnet_b0":
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    return model

# --- Main block for testing this script directly ---
if __name__ == '__main__':
    print("=" * 60)
    print("PneumoVisionAI - Testing Models Module")
    print("=" * 60)
    
    # --- Test SimpleCNN ---
    print("\n--- Testing SimpleCNN ---")
    simple_model = SimpleCNN(num_classes=2)
    dummy_input = torch.randn(4, 3, 224, 224) # (batch_size, channels, height, width)
    output = simple_model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # Expected: (4, 2)
    assert output.shape == (4, 2), "SimpleCNN output shape is incorrect!"
    print("SimpleCNN test passed! ✅")

    # --- Test Pre-trained Model Loader ---
    print("\n--- Testing get_pretrained_model (ResNet50) ---")
    resnet_model = get_pretrained_model("resnet50", num_classes=2, freeze_layers=True)
    
    # Check if layers are frozen
    frozen_params = [p.requires_grad for p in resnet_model.parameters()]
    print(f"Are all base layers frozen? {not any(frozen_params[:-2])}") # Check all but last 2 params (fc layer)
    
    output = resnet_model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # Expected: (4, 2)
    assert output.shape == (4, 2), "ResNet50 output shape is incorrect!"
    print("Pre-trained model test passed! ✅")