# check_val_data.py
import os
from pathlib import Path

def check_validation_data():
    val_path = Path('./data/val')
    
    print("Checking validation data in detail...")
    print("=" * 50)
    
    for class_name in ['NORMAL', 'PNEUMONIA']:
        class_path = val_path / class_name
        
        if class_path.exists():
            # Get all files
            all_files = list(class_path.iterdir())
            
            # Filter image files
            image_extensions = {'.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG'}
            image_files = [f for f in all_files if f.suffix in image_extensions]
            
            # Check for hidden files or other files
            non_image_files = [f for f in all_files if f.suffix not in image_extensions]
            
            print(f"\n{class_name}:")
            print(f"  Total files: {len(all_files)}")
            print(f"  Image files: {len(image_files)}")
            
            # List actual filenames
            print(f"  Image filenames:")
            for img in sorted(image_files)[:10]:  # Show first 10
                print(f"    - {img.name}")
            
            if len(non_image_files) > 0:
                print(f"  Non-image files found:")
                for f in non_image_files:
                    print(f"    - {f.name} (might be hidden file or .DS_Store)")
            
            # Count by extension
            extension_counts = {}
            for img in image_files:
                ext = img.suffix.lower()
                extension_counts[ext] = extension_counts.get(ext, 0) + 1
            
            print(f"  By extension: {extension_counts}")
        else:
            print(f"\n{class_name}: Directory not found!")
    
    # Also check using os.listdir for comparison
    print("\n" + "=" * 50)
    print("Double-checking with os.listdir:")
    
    for class_name in ['NORMAL', 'PNEUMONIA']:
        path = f'./data/val/{class_name}'
        if os.path.exists(path):
            files = os.listdir(path)
            # Filter out hidden files
            visible_files = [f for f in files if not f.startswith('.')]
            print(f"{class_name}: {len(visible_files)} visible files")

if __name__ == "__main__":
    check_validation_data()