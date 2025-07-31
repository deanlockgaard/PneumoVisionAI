# src/train.py
"""
PneumoVisionAI - PyTorch Training Script
Robust, flexible, and user-friendly workflow.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import argparse
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Import our custom modules
from dataset import create_data_loaders
from models import get_pretrained_model, SimpleCNN

class Trainer:
    """A class to encapsulate the complete training and validation workflow."""
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device, writer, model_name, output_dir):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.writer = writer
        self.model_name = model_name
        self.output_dir = output_dir

    def _train_one_epoch(self):
        """Runs a single training epoch."""
        self.model.train()
        total_loss, correct_preds, total_samples = 0.0, 0, 0
        
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        return total_loss / total_samples, correct_preds.float() / total_samples * 100

    def _validate_one_epoch(self):
        """Runs a single validation epoch."""
        self.model.eval()
        total_loss, correct_preds, total_samples = 0.0, 0, 0
        
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_preds += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)
                
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / total_samples, correct_preds.float() / total_samples * 100

    def _plot_history(self, history):
        """Plots and saves the training history."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        epochs_range = range(1, len(history['train_loss']) + 1)
        
        # Plot Loss
        axes[0].plot(epochs_range, history['train_loss'], 'b-o', label='Training Loss')
        axes[0].plot(epochs_range, history['val_loss'], 'r-o', label='Validation Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].legend()

        # Plot Accuracy
        axes[1].plot(epochs_range, history['train_acc'], 'b-o', label='Training Accuracy')
        axes[1].plot(epochs_range, history['val_acc'], 'r-o', label='Validation Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f'{self.model_name}_training_history.png')
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
        plt.show()

    def train(self, num_epochs, patience):
        """Main training loop with checkpointing, early stopping, and plotting."""
        print("🚀 Starting training...")
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_path = os.path.join(self.output_dir, f'best_{self.model_name}.pth')

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self._train_one_epoch()
            val_loss, val_acc = self._validate_one_epoch()

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc.cpu())
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc.cpu())
            
            self.writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
            self.writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)

            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

            if self.scheduler:
                self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                print(f"✅ Validation loss improved ({best_val_loss:.4f} -> {val_loss:.4f}). Saving model...")
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), best_model_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"🛑 Early stopping triggered after {patience} epochs with no improvement.")
                break
                
        print(f"✅ Training complete. Best model saved to {best_model_path}")
        self.writer.close()
        self._plot_history(history)
        
        # Load best model weights back
        self.model.load_state_dict(torch.load(best_model_path))
        return history


def main(args):
    """Main function to set up and run the training process."""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.log_dir, f"{args.model_name}_{timestamp}")
    writer = SummaryWriter(log_dir)
    
    output_dir = './models'
    os.makedirs(output_dir, exist_ok=True)

    train_loader, val_loader, _, info = create_data_loaders(batch_size=args.batch_size, augmentation_level=args.aug_level)
    
    model = SimpleCNN(num_classes=info['num_classes']) if args.model_name == 'simple_cnn' else get_pretrained_model(model_name=args.model_name, num_classes=info['num_classes'])
    
    class_weights_dict = info['class_weights']
    class_weights_tensor = torch.tensor([class_weights_dict[i] for i in sorted(class_weights_dict.keys())], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args.patience // 2, factor=0.2)
    
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, scheduler, device, writer, args.model_name, output_dir)
    trainer.train(args.epochs, args.patience)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PneumoVisionAI model.")
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs.')
    parser.add_argument('--patience', type=int, default=7, help='Patience for early stopping.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--model-name', type=str, default='resnet50', help='Model to use (e.g., resnet50, simple_cnn).')
    parser.add_argument('--aug-level', type=str, default='medium', help='Augmentation level (light, medium, heavy).')
    parser.add_argument('--log-dir', type=str, default='./logs', help='Directory for TensorBoard logs.')
    
    args = parser.parse_args()
    main(args)