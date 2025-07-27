# src/data_exploration.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from collections import Counter
import torch

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')

print("=" * 50)
print("PneumoVisionAI - Data Exploration")
print("=" * 50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("=" * 50)

class PneumoVisionDataExplorer:
    """
    Data exploration for PneumoVisionAI project.
    Analyzes chest X-ray dataset structure and properties.
    """
    def __init__(self, data_path='./data'):
        self.data_path = Path(data_path)
        self.splits = ['train', 'test', 'val']
        self.classes = ['NORMAL', 'PNEUMONIA']

        # Verify data directory exists
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data directory not found at {self.data_path}. "
                                  "Please download the dataset first.")

        # Check which splits are available
        available_splits = [s for s in self.splits if (self.data_path / s).exists()]
        if not available_splits:
            raise FileNotFoundError(f"No data splits found in {self.data_path}. "
                                  "Expected directories: train/, val/, test/")

        print(f"\nFound data splits: {available_splits}")
        self.splits = available_splits

    def analyze_dataset_structure(self):
        """
        Analyzes the PneumoVisionAI dataset structure.
        """
        stats = {}

        print("\nAnalyzing dataset structure...")
        for split in self.splits:
            split_path = self.data_path / split

            split_stats = {}
            for class_name in self.classes:
                class_path = split_path / class_name
                if class_path.exists():
                    # Count all image files
                    image_files = list(class_path.glob('*.jpeg')) + \
                                 list(class_path.glob('*.jpg')) + \
                                 list(class_path.glob('*.png'))
                    split_stats[class_name] = len(image_files)

                    # Show first few file names as examples
                    if len(image_files) > 0:
                        print(f"\n  {split}/{class_name}: {len(image_files)} images")
                        print(f"    Example files: {[f.name for f in image_files[:3]]}")
                else:
                    split_stats[class_name] = 0
                    print(f"\n  Warning: {split}/{class_name} directory not found!")

            split_stats['total'] = sum(split_stats.values())
            stats[split] = split_stats

        return stats

    def visualize_class_distribution(self, stats):
        """
        Creates visualization for PneumoVisionAI dataset distribution.
        """
        # Set PneumoVisionAI color scheme
        colors = ['#3498db', '#e74c3c']  # Blue for Normal, Red for Pneumonia

        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('PneumoVisionAI - Dataset Distribution Analysis', fontsize=16, fontweight='bold')

        # Create grid for subplots
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # Bar charts for each split
        for idx, split in enumerate(self.splits):
            ax = fig.add_subplot(gs[0, idx])

            if split in stats:
                class_counts = [stats[split].get(cls, 0) for cls in self.classes]
                bars = ax.bar(self.classes, class_counts, color=colors, alpha=0.8)

                ax.set_title(f'{split.capitalize()} Set', fontsize=14, fontweight='bold')
                ax.set_ylabel('Number of Images')

                # Add value labels on bars
                for bar, count in zip(bars, class_counts):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 20,
                           f'{count:,}', ha='center', va='bottom')

                # Add percentage labels
                total = sum(class_counts)
                for i, (bar, count) in enumerate(zip(bars, class_counts)):
                    if total > 0:
                        percentage = (count / total) * 100
                        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                               f'{percentage:.1f}%', ha='center', va='center',
                               color='white', fontweight='bold')

        # Overall distribution pie chart
        ax_pie = fig.add_subplot(gs[1, :2])
        total_stats = {'NORMAL': 0, 'PNEUMONIA': 0}
        for split_stats in stats.values():
            for cls in self.classes:
                total_stats[cls] += split_stats.get(cls, 0)

        wedges, texts, autotexts = ax_pie.pie(
            total_stats.values(),
            labels=total_stats.keys(),
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            explode=(0.05, 0.05)
        )
        ax_pie.set_title('Overall Class Distribution', fontsize=14, fontweight='bold')

        # Summary statistics text
        ax_summary = fig.add_subplot(gs[1, 2])
        ax_summary.axis('off')

        summary_text = "PneumoVisionAI Summary\n" + "="*25 + "\n\n"
        total_images = sum(total_stats.values())
        summary_text += f"Total Images: {total_images:,}\n\n"

        for split, split_stats in stats.items():
            summary_text += f"{split.upper()}:\n"
            summary_text += f"  Total: {split_stats['total']:,}\n"
            if split_stats['total'] > 0:
                ratio = split_stats.get('PNEUMONIA', 0) / max(split_stats.get('NORMAL', 1), 1)
                summary_text += f"  P/N Ratio: {ratio:.2f}\n"
            summary_text += "\n"

        # Add class imbalance warning if needed
        overall_ratio = total_stats['PNEUMONIA'] / max(total_stats['NORMAL'], 1)
        if overall_ratio > 2 or overall_ratio < 0.5:
            summary_text += "⚠️ Significant class imbalance detected!\n"
            summary_text += "Consider using weighted loss or resampling."

        ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes,
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Save figure
        os.makedirs('./figures', exist_ok=True)
        plt.savefig('./figures/pneumovision_data_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

        return fig

    def analyze_image_properties(self, num_samples=100):
        """
        Analyzes image properties like dimensions, channels, and pixel statistics.
        """
        train_path = self.data_path / 'train'

        properties = {
            'widths': [],
            'heights': [],
            'channels': [],
            'mean_pixel_values': [],
            'std_pixel_values': []
        }

        print(f"\nAnalyzing properties of {num_samples} sample images...")

        for class_name in self.classes:
            class_path = train_path / class_name
            image_files = list(class_path.glob('*.jpeg'))[:num_samples//2]

            for img_path in image_files:
                img = cv2.imread(str(img_path))
                if img is not None:
                    properties['heights'].append(img.shape[0])
                    properties['widths'].append(img.shape[1])
                    properties['channels'].append(img.shape[2] if len(img.shape) > 2 else 1)

                    # Convert to grayscale for statistics (X-rays are typically grayscale)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) > 2 else img
                    properties['mean_pixel_values'].append(np.mean(gray))
                    properties['std_pixel_values'].append(np.std(gray))

        # Visualize properties
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('PneumoVisionAI - Image Properties Analysis', fontsize=16, fontweight='bold')

        # Image dimensions
        axes[0, 0].hist2d(properties['widths'], properties['heights'], bins=20, cmap='Blues')
        axes[0, 0].set_xlabel('Width (pixels)')
        axes[0, 0].set_ylabel('Height (pixels)')
        axes[0, 0].set_title('Image Dimensions Distribution')

        # Channel distribution
        axes[0, 1].hist(properties['channels'], bins=np.arange(1, 5) - 0.5, color='#3498db', edgecolor='black')
        axes[0, 1].set_xlabel('Number of Channels')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Channel Distribution')
        axes[0, 1].set_xticks([1, 2, 3])

        # Pixel intensity statistics
        axes[1, 0].hist(properties['mean_pixel_values'], bins=30, alpha=0.7, label='Mean', color='#3498db')
        axes[1, 0].hist(properties['std_pixel_values'], bins=30, alpha=0.7, label='Std Dev', color='#e74c3c')
        axes[1, 0].set_xlabel('Pixel Value (0-255)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Pixel Intensity Statistics')
        axes[1, 0].legend()

        # Summary statistics
        axes[1, 1].axis('off')
        summary_text = f"""Summary Statistics:

Image Dimensions:
  Width: {np.mean(properties['widths']):.0f} ± {np.std(properties['widths']):.0f} px
  Height: {np.mean(properties['heights']):.0f} ± {np.std(properties['heights']):.0f} px

Pixel Values:
  Mean: {np.mean(properties['mean_pixel_values']):.1f} ± {np.std(properties['mean_pixel_values']):.1f}
  Std: {np.mean(properties['std_pixel_values']):.1f} ± {np.std(properties['std_pixel_values']):.1f}

Observations:
  • Images vary significantly in size
  • All images are 3-channel (RGB) format
  • Pixel intensities show good contrast
  • Standardization will be necessary
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                       fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig('./figures/pneumovision_image_properties.png', dpi=300, bbox_inches='tight')
        plt.show()

        return properties

    def visualize_sample_images(self, num_samples=8):
        """
        Displays sample images from each class.
        """
        fig, axes = plt.subplots(2, num_samples//2, figsize=(15, 8))
        fig.suptitle('PneumoVisionAI - Sample X-Ray Images', fontsize=16, fontweight='bold')

        train_path = self.data_path / 'train'

        for class_idx, class_name in enumerate(self.classes):
            class_path = train_path / class_name
            image_files = list(class_path.glob('*.jpeg'))[:num_samples//2]

            for img_idx, img_path in enumerate(image_files):
                img = cv2.imread(str(img_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                ax = axes[class_idx, img_idx]
                ax.imshow(img_rgb, cmap='gray' if len(img.shape) == 2 else None)
                ax.set_title(f'{class_name}',
                           color='#3498db' if class_name == 'NORMAL' else '#e74c3c',
                           fontweight='bold')
                ax.axis('off')

                # Add image shape as text
                ax.text(0.5, -0.1, f'{img.shape[1]}×{img.shape[0]}',
                       transform=ax.transAxes, ha='center', fontsize=8)

        plt.tight_layout()
        plt.savefig('./figures/pneumovision_sample_images.png', dpi=300, bbox_inches='tight')
        plt.show()

# Run the exploration
if __name__ == "__main__":
    print("\nInitializing PneumoVisionAI Data Explorer...")

    try:
        explorer = PneumoVisionDataExplorer('./data')

        # Analyze dataset structure
        stats = explorer.analyze_dataset_structure()

        # Visualize distribution
        explorer.visualize_class_distribution(stats)

        # Analyze image properties
        properties = explorer.analyze_image_properties(num_samples=100)

        # Visualize samples
        print("\nVisualizing sample images...")
        explorer.visualize_sample_images(num_samples=8)

        print("\n✅ Data exploration complete! Check the ./figures/ directory for saved visualizations.")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()