
# 🫁 PneumoVisionAI

**AI-Powered Pneumonia Detection from Chest X-rays**

A deep learning project using PyTorch to classify chest X-ray images for pneumonia detection. This project demonstrates computer vision techniques, medical image analysis, and responsible AI practices in healthcare applications.

---

## 🎯 Project Overview

PneumoVisionAI uses convolutional neural networks (CNNs) to automatically detect pneumonia in chest X-ray images. The system is trained on the [Kaggle Chest X-ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset and achieves high accuracy in distinguishing between normal and pneumonia-affected X-rays.

---

## ⚠️ Medical Data Privacy

**DO NOT COMMIT MEDICAL DATA TO GIT**

This project works with medical imaging data. Please:
- Never commit patient data or medical images to version control
- Keep all data in the `./data/` directory (which is gitignored)
- Follow HIPAA guidelines if working with real patient data
- Use only publicly available research datasets

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/PneumoVisionAI.git
cd PneumoVisionAI
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
```bash
# Requires Kaggle API credentials (~1.15GB)
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d ./data/
rm chest-xray-pneumonia.zip  # Clean up zip file
```

### 5. Verify Data Structure
Your `./data/` directory should look like:

```
./data/
├── train/
│   ├── NORMAL/       # 1,341 images
│   └── PNEUMONIA/    # 3,875 images
├── val/
│   ├── NORMAL/       # 8 images
│   └── PNEUMONIA/    # 8 images
└── test/
    ├── NORMAL/       # 234 images
    └── PNEUMONIA/    # 390 images
```

---

## 📁 Project Structure

```
PneumoVisionAI/
├── src/
│   ├── data_exploration.py   # Dataset analysis and visualization
│   ├── dataset.py            # PyTorch dataset and data loaders
│   ├── models.py             # CNN architectures
│   ├── train.py              # Training pipeline
│   └── evaluate.py           # Model evaluation and metrics
├── notebooks/
│   └── exploration.ipynb     # Jupyter notebooks for analysis
├── models/                   # Saved model checkpoints (gitignored)
├── logs/                     # Training logs (gitignored)
├── figures/                  # Generated visualizations
├── data/                     # Dataset directory (gitignored)
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore file
└── README.md                 # This file
```

---

## 🔧 Technologies Used

- **PyTorch** – Deep learning framework  
- **Albumentations** – Advanced image augmentation  
- **OpenCV** – Image processing  
- **Matplotlib / Seaborn** – Visualization  
- **NumPy / Pandas** – Data manipulation  
- **TensorBoard** – Training visualization  

---

## 📊 Model Performance

> _To be updated after training_

- **Accuracy:**  
- **Precision:**  
- **Recall:**  
- **F1-Score:**  
- **AUC-ROC:**  

---

## 🏃‍♂️ Training

### Default training
```bash
python src/train.py
```

### Custom parameters
```bash
python src/train.py --epochs 50 --batch-size 32 --learning-rate 0.001
```

---

## 📈 Monitoring

Visualize training progress with TensorBoard:

```bash
tensorboard --logdir logs/
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request.

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) on Kaggle  
- **Original Source**:  
  Kermany et al. “Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.” _Cell_, 2018.

---

> _Disclaimer: This is an educational project. Any model developed here should not be used for actual medical diagnosis without proper validation and regulatory approval._
