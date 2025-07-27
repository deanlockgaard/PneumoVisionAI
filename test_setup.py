# test_setup.py
import os
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Check data directory
data_exists = os.path.exists('./data/train')
print(f"Data directory exists: {data_exists}")

image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff")

if data_exists:
    for split in ['train', 'val', 'test']:
        if os.path.exists(f'./data/{split}'):
            normal_files = [
                f for f in os.listdir(f'./data/{split}/NORMAL')
                if os.path.isfile(os.path.join(f'./data/{split}/NORMAL', f)) and f.lower().endswith(image_extensions)
            ]

            pneumonia_files = [
                f for f in os.listdir(f'./data/{split}/PNEUMONIA')
                if os.path.isfile(os.path.join(f'./data/{split}/PNEUMONIA', f)) and f.lower().endswith(image_extensions)
            ]

            normal = len(normal_files)
            pneumonia = len(pneumonia_files)

            print(f"{split}: {normal} normal, {pneumonia} pneumonia")