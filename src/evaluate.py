# src/evaluate.py
"""
PneumoVisionAI - Final Evaluation Script
Loads the best model, evaluates on the test set, and generates comprehensive metrics and visuals.
"""
import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from tqdm import tqdm

from dataset import create_data_loaders
from models import get_pretrained_model, SimpleCNN

def evaluate_model(model, data_loader, device):
    """Runs inference and returns true labels, predictions, and probability for class 1."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating on Test Set"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability for PNEUMONIA
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def plot_confusion(labels, preds, class_names, save_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Test Set)')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.show()

def plot_roc(labels, probs, save_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}', color='darkorange')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Test Set)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"ROC curve saved to {save_path}")
    plt.show()

def print_and_save_metrics(y_true, y_pred, y_prob, class_names, save_path=None):
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    acc = np.mean(y_true == y_pred)
    cm = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    # Sensitivity (Recall for class 1 - Pneumonia)
    sensitivity = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
    # Specificity (Recall for class 0 - Normal)
    specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
    text = (
        "\n--- Classification Report (Test Set) ---\n" + report +
        f"\nAccuracy: {acc:.4f}"
        f"\n\nConfusion Matrix:\n{cm}"
        f"\n\n--- Key Clinical Metrics ---"
        f"\nSensitivity (Pneumonia Recall): {sensitivity:.4f}"
        f"\nSpecificity (Normal Recall):   {specificity:.4f}"
        f"\nROC-AUC:                       {auc:.4f}\n"
    )
    print(text)
    if save_path:
        with open(save_path, 'w') as f:
            f.write(text)
        print(f"Metrics summary saved to {save_path}")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load test set, using only preprocessing (no random augmentation)
    _, _, test_loader, info = create_data_loaders(batch_size=args.batch_size, augmentation_level='light', num_workers=0)

    # Class names: dynamic, safe fallback provided
    class_names = info.get('classes', ['NORMAL', 'PNEUMONIA'])

    # Model selection: CLI override, fallback to filename
    if args.model_arch:
        model_arch = args.model_arch.lower()
    else:
        base = os.path.basename(args.model_path)
        model_arch = 'simple_cnn' if 'simple_cnn' in base else 'resnet50'
    if model_arch == 'simple_cnn':
        model = SimpleCNN(num_classes=len(class_names))
    else:
        model = get_pretrained_model(model_arch, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    print(f"Loaded {model_arch} model weights from {args.model_path}")

    # Evaluate
    y_true, y_pred, y_prob = evaluate_model(model, test_loader, device)

    # Metrics report and save
    metrics_save = os.path.join(args.output_dir, f'{model_arch}_test_metrics.txt')
    print_and_save_metrics(y_true, y_pred, y_prob, class_names, save_path=metrics_save)
    # Plots
    cm_fig = os.path.join(args.output_dir, f'{model_arch}_confusion_matrix.png')
    roc_fig = os.path.join(args.output_dir, f'{model_arch}_roc_curve.png')
    plot_confusion(y_true, y_pred, class_names, cm_fig)
    plot_roc(y_true, y_prob, roc_fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PneumoVisionAI model on test set.")
    parser.add_argument('--model-path', type=str, required=True, help='Path to .pth model file.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation.')
    parser.add_argument('--model-arch', type=str, default=None, help='Model architecture (e.g., resnet50, simple_cnn).')
    parser.add_argument('--output-dir', type=str, default='./figures', help='Directory to save figures/metrics.')
    args = parser.parse_args()
    main(args)
