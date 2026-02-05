# 🧠 Brain Tumor Detection Project

A comprehensive machine learning project for brain tumor classification and segmentation using deep learning techniques. This project implements multiple approaches for detecting and analyzing brain tumors from MRI scans.

## 📋 Project Overview

This project provides three complementary approaches to brain tumor analysis:

1. **Classification** - Categorize tumor types (glioma, pituitary, meningioma, no tumor)
2. **2D Segmentation** - U-Net architecture for pixel-wise tumor localization in 2D slices
3. **3D Segmentation** - Volumetric analysis of 3D MRI data (BRaTS 2023)

## 🎯 Features

### Classification Model
- Multi-class tumor type classification
- Supports 4 categories: glioma, pituitary, meningioma, no tumor
- CNN-based architecture with transfer learning capabilities
- Automated data preprocessing and augmentation

### 2D Segmentation (U-Net)
- U-Net encoder-decoder architecture
- Custom data generator with CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Dice coefficient and Dice loss metrics
- Binary segmentation for tumor localization
- Compatible with TIF, JPG, JPEG, PNG formats

### 3D Segmentation
- Processing of volumetric MRI data (BRaTS 2023 dataset)
- Multi-modal support (T1, T1c, T2w, T2 FLAIR)
- 3D CNN architecture
- Target shape: (160, 192, 160)

### API Deployment
- FastAPI endpoints for model inference
- RESTful API for integration with other applications
- Docker support for containerized deployment

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.11+** - Main programming language
- **TensorFlow 2.19.0** - Deep learning framework
- **Keras** - High-level neural networks API
- **FastAPI** - Modern web framework for APIs
- **OpenCV (cv2)** - Image processing

### Data Processing
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning utilities
- **Matplotlib** - Visualization
- **Seaborn** - Statistical data visualization

### Cloud & Storage
- **Google Cloud Storage** - Data storage
- **Google BigQuery** - Data warehousing
- **Google Cloud Platform** - Cloud infrastructure

### Development Tools
- **Docker** - Containerization
- **Jupyter Notebooks** - Interactive development
- **pytest** - Testing framework
- **Makefile** - Build automation

## 📊 Datasets

### Classification Dataset
- **Source**: Kaggle Brain MRI Dataset
- **Classes**: 4 tumor types
- **Format**: Images with associated labels
- **Splits**: Training/Testing directories

### 2D Segmentation Dataset
- **Source**: [Brain MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Format**: TIF images with corresponding mask files
- **Content**: MRI scans + binary tumor masks
- **Image Size**: 256x256x3 (RGB)

### 3D Segmentation Dataset
- **Source**: BRaTS 2023 Challenge
- **Modalities**: T1, T1c, T2w, T2 FLAIR
- **Format**: NIfTI files (.nii.gz)
- **Target Shape**: 160x192x160

## 🏗️ Project Structure

```
brain_tumor_detection_project/
├── brain/
│   ├── api/                    # FastAPI endpoints
│   │   ├── fast.py           # Production API
│   │   └── fast_local.py     # Local development API
│   ├── interface/              # Main entry point and CLI
│   │   ├── main.py           # Training/evaluation orchestration
│   │   └── __init__.py
│   ├── ml_logic_classification/  # Classification model logic
│   │   ├── preprocess.py     # Data preprocessing
│   │   ├── utils.py          # Utility functions
│   │   ├── encoders.py       # Image encoders
│   │   └── model.py         # Classification model
│   ├── ml_logic_segmentation_2D/  # 2D segmentation
│   │   ├── preprocess.py     # Data preprocessing
│   │   ├── utils.py          # Utility functions
│   │   ├── metrics.py        # Dice metrics
│   │   ├── data.py           # Data generators
│   │   └── model.py         # U-Net model
│   ├── ml_logic_segmentation_3D/  # 3D segmentation
│   │   ├── preprocess.py     # 3D data preprocessing
│   │   ├── utils.py          # Utility functions
│   │   ├── metrics.py        # 3D metrics
│   │   ├── data.py           # 3D data handling
│   │   ├── volumes.py        # Volume processing
│   │   └── model.py         # 3D model
│   ├── params.py              # Configuration parameters
│   ├── registry.py           # Model registry
│   └── __init__.py
├── notebooks/                # Jupyter notebooks
│   ├── EDA_Victor/         # Exploratory data analysis
│   ├── bt_2023_preprocessing.ipynb
│   ├── bt_2023_segmentation.ipynb
│   └── ...
├── raw_data/                # Raw data storage
│   ├── classification/       # Classification images
│   └── segmentation/        # Segmentation datasets
├── requirements.txt          # Python dependencies
├── setup.py                 # Package setup
├── Makefile                 # Build automation
├── Dockerfile              # Docker configuration
└── .envrc                 # Environment configuration
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Docker (optional, for containerized deployment)
- Google Cloud account (for cloud storage)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/DaLiryc/brain_tumor_detection_project.git
cd brain_tumor_detection_project
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install the package**
```bash
make reinstall_package
```

### Environment Configuration

Create a `.env` file with the following variables:

```bash
# Google Cloud Configuration
BUCKET_NAME=your-bucket-name
GCP_PROJECT=your-project-id
GCP_REGION=your-region

# Model Configuration
MODEL_TARGET=local  # or 'gcs' for cloud storage
DATA_TARGET=gcs     # or 'local' for local storage

# Paths (adjust as needed)
TRAIN_DIR=/path/to/training/data
TEST_DIR=/path/to/testing/data
```

### Running the Models

#### Classification

```bash
# Preprocess data
make run_preprocess_classification

# Train model
make run_train_classification

# Evaluate model
make run_evaluate_classification

# Run full pipeline
make run_all_classification
```

#### 2D Segmentation

```bash
# Preprocess data
make run_preprocess_seg2D

# Train model
make run_train_seg2D

# Evaluate model
make run_evaluate_seg2D

# Run full pipeline
make run_all_seg2D
```

#### 3D Segmentation

```bash
# Preprocess data
make run_preprocess_seg3D
```

### Running the API

```bash
# Start local development server
make run_api_local

# Start production server
make run_api
```

The API will be available at `http://localhost:8000`

## 📈 Model Performance

### Classification Metrics
- Accuracy: TBD (depends on training)
- F1-Score: TBD
- Confusion Matrix: Available in evaluation notebooks

### 2D Segmentation Metrics
- **Dice Coefficient**: Measures overlap between predicted and actual masks
- **Dice Loss**: Complementary loss metric
- Visual inspection through prediction plots

### 3D Segmentation Metrics
- Volume-based metrics
- Multi-modal evaluation across MRI sequences

## 🔧 Configuration

### Classification Parameters
- `TARGET_SIZE`: (224, 224) - Input image size
- `BATCH_SIZE`: 32 - Training batch size
- `EPOCHS_CLASS`: 2 - Number of training epochs (adjust for production)
- `CLASSES`: ["glioma", "pituitary", "meningioma", "notumor"]

### 2D Segmentation Parameters
- `IMG_SIZE`: 256 - Input image dimensions
- `EPOCHS_SEG2D`: 50 - Number of training epochs
- `IMG_CHANNELS`: 3 - RGB channels
- Model: U-Net with ~1M parameters

### 3D Segmentation Parameters
- `TARGET_SHAPE_3D`: (160, 192, 160) - Volume dimensions
- `MODALITIES_3D`: T1, T1c, T2w, T2 FLAIR
- Format: NIfTI files (.nii.gz)

## 🧪 Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=brain --cov-report=html
```

## 🐳 Docker Deployment

### Build the image

```bash
docker build -t brain-tumor-detection .
```

### Run the container

```bash
docker run -p 8000:8000 \
  -e BUCKET_NAME=your-bucket \
  -e GCP_PROJECT=your-project \
  brain-tumor-detection
```

## 📚 Documentation

- **2D Segmentation Details**: See `README_SEG2D.md` for detailed U-Net implementation
- **Notebooks**: Comprehensive EDA and model training in `notebooks/` directory
- **API Documentation**: Access at `/docs` endpoint when API is running

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Kaggle Brain MRI Dataset** - For providing the classification and 2D segmentation data
- **BRaTS Challenge** - For the 3D segmentation benchmark dataset
- **Le Wagon Brainers** - Original project team
- **TensorFlow/Keras Community** - For excellent deep learning tools

## ⚠️ Disclaimer

This project is for educational and research purposes only. The models and predictions should not be used for actual medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.

## 🎯 Future Enhancements

See [`next_steps.md`](./next_steps.md) for planned improvements and development roadmap.
