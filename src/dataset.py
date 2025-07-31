# src/dataset.py
"""
PneumoVisionAI - Dataset Module
Handles data loading, preprocessing, and augmentation for chest X-ray images.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Optional, Dict, List
import matplotlib.pyplot as plt
from tqdm import tqdm

class PneumoVisionDataset(Dataset):
    """
    Custom PyTorch Dataset for PneumoVisionAI project.
    
    This class handles:
    - Loading chest X-ray images from disk
    - Applying data augmentations
    - Converting images to PyTorch tensors
    - Managing class labels and mappings
    """
    
    def __init__(
        self, 
        root_dir: str = './data', 
        split: str = 'train', 
        transform=None, 
        use_albumentations: bool = True,
        cache_images: bool = False
    ):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Path to data directory containing train/val/test folders
            split: One of 'train', 'val', or 'test'
            transform: Optional transform to apply to images
            use_albumentations: Whether to use albumentations (True) or torchvision transforms
            cache_images: Whether to cache all images in memory (faster but uses more RAM)
        """
        self.root_dir = Path(root_dir) / split
        self.use_albumentations = use_albumentations
        self.transform = transform
        self.cache_images = cache_images
        
        # Verify directory exists
        if not self.root_dir.exists():
            raise ValueError(f"Directory not found: {self.root_dir}")
        
        # Class mappings
        self.class_to_idx = {'NORMAL': 0, 'PNEUMONIA': 1}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.classes = list(self.class_to_idx.keys())
        
        # Collect all image paths and labels
        self.image_paths = []
        self.labels = []
        self.image_cache = {}
        
        # Load all image paths
        for class_name, class_idx in self.class_to_idx.items():
            class_path = self.root_dir / class_name
            if class_path.exists():
                # Support multiple image formats
                for ext in ['*.jpeg', '*.jpg', '*.png']:
                    for img_path in class_path.glob(ext):
                        self.image_paths.append(img_path)
                        self.labels.append(class_idx)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.root_dir}")
        
        print(f"[PneumoVisionAI] Loaded {len(self.image_paths)} images from {split} set")
        print(f"  - Normal: {self.labels.count(0)}")
        print(f"  - Pneumonia: {self.labels.count(1)}")
        
        # Cache images if requested
        if cache_images:
            print(f"[PneumoVisionAI] Caching {split} images in memory...")
            for idx in tqdm(range(len(self.image_paths))):
                img_path = self.image_paths[idx]
                image = cv2.imread(str(img_path))
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self.image_cache[idx] = image
    
    def __len__(self) -> int:
        """Return the total number of images."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            tuple: (image_tensor, label)
        """
        # Load image
        if self.cache_images and idx in self.image_cache:
            image = self.image_cache[idx].copy()  # Copy to avoid modifying cache
        else:
            img_path = self.image_paths[idx]
            image = cv2.imread(str(img_path))
            
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get label
        label = self.labels[idx]
        
        # Apply transformations
        if self.transform:
            if self.use_albumentations:
                # Albumentations expects image as numpy array
                transformed = self.transform(image=image)
                image = transformed['image']
            else:
                # Torchvision transforms expect PIL Image
                image = Image.fromarray(image)
                image = self.transform(image)
        
        return image, label
    
    def get_sample_weights(self) -> np.ndarray:
        """
        Calculate sample weights for dealing with class imbalance.
        Used for WeightedRandomSampler in PyTorch.
        
        Returns:
            numpy array of weights for each sample
        """
        class_counts = np.bincount(self.labels)
        class_weights = 1. / class_counts
        sample_weights = class_weights[self.labels]
        return sample_weights
    
    def get_class_weights(self) -> Dict[int, float]:
        """
        Calculate class weights for use in loss function.
        
        Returns:
            Dictionary mapping class indices to weights
        """
        class_counts = np.bincount(self.labels)
        total_samples = len(self.labels)
        n_classes = len(class_counts)
        
        # Calculate balanced class weights
        class_weights = {}
        for i in range(n_classes):
            class_weights[i] = total_samples / (n_classes * class_counts[i])
            
        return class_weights


def get_transforms(
    img_size: int = 224, 
    augmentation_level: str = 'medium',
    mean: List[float] = None,
    std: List[float] = None
) -> Tuple[A.Compose, A.Compose]:
    """
    Create transformation pipelines for training and validation.
    
    Args:
        img_size: Target image size (square)
        augmentation_level: 'light', 'medium', or 'heavy'
        mean: Normalization mean values (default: ImageNet)
        std: Normalization std values (default: ImageNet)
        
    Returns:
        tuple: (train_transform, val_transform)
    """
    
    # Default normalization parameters (ImageNet)
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    # Base transformations (always applied)
    base_transforms = [
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
    ]
    
    # Augmentation transforms based on level
    if augmentation_level == 'light':
        aug_transforms = [
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
        ]
    
    elif augmentation_level == 'medium':
        aug_transforms = [
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.GaussianBlur(blur_limit=(3, 7)),
            ], p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT
            ),
        ]
    
    else:  # heavy
        aug_transforms = [
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=7),
            ], p=0.7),
            
            # Geometric distortions
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.5),
                A.GridDistortion(distort_limit=0.5),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50),
            ], p=0.3),
            
            # Brightness and contrast
            A.OneOf([
                A.CLAHE(clip_limit=4.0),
                A.Equalize(),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3
                ),
                A.RandomGamma(gamma_limit=(80, 120)),
            ], p=0.5),
            
            # Affine transformations
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=20,
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT
            ),
            
            # Cutout / Coarse Dropout
            A.CoarseDropout(
                max_holes=8,
                max_height=img_size//8,
                max_width=img_size//8,
                min_holes=1,
                min_height=img_size//16,
                min_width=img_size//16,
                fill_value=0,
                p=0.3
            ),
        ]
    
    # Final transforms (always applied)
    final_transforms = [
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ]
    
    # Compose transforms
    train_transform = A.Compose(base_transforms + aug_transforms + final_transforms)
    val_transform = A.Compose(base_transforms + final_transforms)
    
    return train_transform, val_transform


def create_data_loaders(
    data_path: str = './data',
    batch_size: int = 32,
    img_size: int = 224,
    augmentation_level: str = 'medium',
    num_workers: int = 0,
    use_weighted_sampling: bool = True,
    cache_images: bool = False,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create PyTorch DataLoaders for train, validation, and test sets.
    
    Args:
        data_path: Path to data directory
        batch_size: Batch size for training
        img_size: Target image size
        augmentation_level: Level of augmentation ('light', 'medium', 'heavy')
        num_workers: Number of parallel workers for data loading
        use_weighted_sampling: Whether to use weighted sampling for class imbalance
        cache_images: Whether to cache images in memory
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, dataset_info)
    """
    
    print(f"\n[PneumoVisionAI] Creating data loaders...")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Image size: {img_size}x{img_size}")
    print(f"  - Augmentation: {augmentation_level}")
    print(f"  - Workers: {num_workers}")
    
    # Get transforms
    train_transform, val_transform = get_transforms(img_size, augmentation_level)
    
    # Create datasets
    train_dataset = PneumoVisionDataset(
        root_dir=data_path,
        split='train',
        transform=train_transform,
        cache_images=cache_images
    )
    
    val_dataset = PneumoVisionDataset(
        root_dir=data_path,
        split='val',
        transform=val_transform,
        cache_images=cache_images
    )
    
    test_dataset = PneumoVisionDataset(
        root_dir=data_path,
        split='test',
        transform=val_transform,
        cache_images=cache_images
    )
    
    # Create weighted sampler for training if requested
    train_sampler = None
    if use_weighted_sampling:
        train_weights = train_dataset.get_sample_weights()
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_weights),
            replacement=True
        )
        print(f"  - Using weighted sampling for class balance")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # Only shuffle if not using sampler
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch for stable training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Collect dataset information
    dataset_info = {
        'num_classes': len(train_dataset.classes),
        'classes': train_dataset.classes,
        'class_to_idx': train_dataset.class_to_idx,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'class_weights': train_dataset.get_class_weights(),
        'img_size': img_size,
        'augmentation_level': augmentation_level
    }
    
    print(f"\n[PneumoVisionAI] Data loaders created successfully!")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, dataset_info


def visualize_batch(
    data_loader: DataLoader,
    num_samples: int = 8,
    denormalize: bool = True,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize a batch of images from a data loader.
    
    Args:
        data_loader: PyTorch DataLoader to visualize from
        num_samples: Number of samples to display
        denormalize: Whether to denormalize images for display
        save_path: Optional path to save the visualization
    """
    # Get a batch
    images, labels = next(iter(data_loader))
    
    # Limit to requested number of samples
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Denormalize if requested
    if denormalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        images = images * std + mean
        images = torch.clamp(images, 0, 1)
    
    # Create figure
    fig, axes = plt.subplots(
        2, 
        num_samples // 2, 
        figsize=(15, 8)
    )
    axes = axes.flatten()
    
    # Plot images
    for i, (img, label) in enumerate(zip(images, labels)):
        # Convert to numpy and transpose for matplotlib
        img_np = img.permute(1, 2, 0).numpy()
        
        axes[i].imshow(img_np)
        axes[i].set_title(
            f'{"PNEUMONIA" if label == 1 else "NORMAL"}',
            color='red' if label == 1 else 'green',
            fontweight='bold'
        )
        axes[i].axis('off')
    
    plt.suptitle('PneumoVisionAI - Sample Batch Visualization', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def compute_dataset_stats(data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and standard deviation of the dataset.
    
    This is useful if you want to use dataset-specific normalization
    instead of ImageNet statistics.
    
    Args:
        data_loader: DataLoader to compute statistics from
        
    Returns:
        tuple: (mean, std) as numpy arrays
    """
    print("Computing dataset statistics...")
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    for data, _ in tqdm(data_loader):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    return mean.numpy(), std.numpy()


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("PneumoVisionAI - Testing Dataset Module")
    print("=" * 60)
    
    # Create data loaders
    train_loader, val_loader, test_loader, dataset_info = create_data_loaders(
        data_path='./data',
        batch_size=16,
        img_size=224,
        augmentation_level='medium',
        num_workers=2
    )
    
    # Print dataset information
    print("\nDataset Information:")
    print("-" * 40)
    for key, value in dataset_info.items():
        if key != 'class_weights':  # Skip the dict for cleaner output
            print(f"{key}: {value}")
    
    # Print class weights
    print("\nClass Weights (for handling imbalance):")
    for class_idx, weight in dataset_info['class_weights'].items():
        class_name = 'NORMAL' if class_idx == 0 else 'PNEUMONIA'
        print(f"  {class_name}: {weight:.3f}")
    
    # Visualize training batch
    print("\nVisualizing training batch (with augmentation)...")
    visualize_batch(
        train_loader, 
        num_samples=8,
        save_path='./figures/train_batch_sample.png'
    )
    
    # Visualize validation batch
    print("\nVisualizing validation batch (no augmentation)...")
    visualize_batch(
        val_loader,
        num_samples=8,
        save_path='./figures/val_batch_sample.png'
    )
    
    # Optional: Compute dataset-specific statistics
    # print("\nComputing dataset statistics...")
    # mean, std = compute_dataset_stats(train_loader)
    # print(f"Dataset mean: {mean}")
    # print(f"Dataset std: {std}")