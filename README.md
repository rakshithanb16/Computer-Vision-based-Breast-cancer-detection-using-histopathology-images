# Computer-Vision-based-Breast-cancer-detection-using-histopathology-images

This project explores deep learning methods for **breast cancer detection** using histopathology images.  
We use an ensemble of DenseNet201, Xception, and GoogLeNet to classify HER2 expression levels in breast cancer histopathology images, evaluated on H&E and IHC datasets.

## 📌 Features
- Ensemble learning for better accuracy
- Grad-CAM visualization for interpretability
- Bilateral filtering preprocessing to reduce noise
- Achieved **91.54% (H&E)** and **94.69% (IHC)** accuracy with 0.7 confidence threshold.

## 📂 Dataset
We used the **Breast Cancer Immunohistochemistry (BCI) dataset** with H&E and IHC stained images.


## 🛠️ Tech Stack
- Programming & Deep Learning: Python 3, TensorFlow/Keras
- Preprocessing & Augmentation: OpenCV (bilateral filtering, resizing, augmentation)
- Data Handling & Analysis: NumPy, Pandas, SciPy, scikit-learn (StratifiedKFold, metrics)
- Visualization: Matplotlib (Grad-CAM overlays, plots)
- Model Architectures: DenseNet201, Xception, InceptionV3 (as GoogLeNet surrogate)
- Utilities: Python standard libraries (os, re, math, random, gc, typing)

## 🚀 Results
- Accuracy: 91.54% (H&E), 94.69% (IHC)
- F1-Score: 0.9117 (H&E), 0.9455 (IHC)




