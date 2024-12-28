# Sound Classifier Project

This project implements a machine learning-based sound classification system that can identify different types of sounds or audio issues. It supports multiple classification algorithms and provides an easy-to-use interface for training and prediction.

## 📋 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)

## ✨ Features
- Support for multiple machine learning models (Random Forest, Logistic Regression, SVM, Neural Network)
- Automatic audio feature extraction using librosa
- Easy-to-use interface for training and prediction
- Cross-platform compatibility (Windows and Mac)
- Comprehensive audio analysis using MFCCs and spectral features

## 📁 Project Structure
```
varun-project/
├── sound_classifier.py   # Main classifier implementation
├── requirements.txt      # Python dependencies
└── .gitignore           # Git ignore file
```

### File Descriptions
- `sound_classifier.py`: Contains the main `SoundClassifier` class that handles feature extraction, model training, and prediction
- `requirements.txt`: Lists all Python package dependencies with their versions
- `.gitignore`: Specifies which files Git should ignore

## 🔧 Prerequisites
Before you begin, ensure you have the following installed:
- Python 3.7 or higher
- pip (Python package installer)

## 📥 Installation

### For Windows:

1. Install Python:
   - Download Python from [python.org](https://www.python.org/downloads/)
   - During installation, make sure to check "Add Python to PATH"
   - Verify installation by opening Command Prompt and typing:
     ```
     python --version
     ```

2. Clone or download this repository:
   ```
   git clone <repository-url>
   ```
   Or download and extract the ZIP file

3. Open Command Prompt as Administrator and navigate to the project directory:
   ```
   cd path\to\varun-project
   ```

4. Create a virtual environment (recommended):
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

5. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### For Mac:

1. Install Python (if not already installed):
   ```
   brew install python
   ```
   Or download from [python.org](https://www.python.org/downloads/)

2. Clone or download this repository:
   ```
   git clone <repository-url>
   ```
   Or download and extract the ZIP file

3. Open Terminal and navigate to the project directory:
   ```
   cd path/to/varun-project
   ```

4. Create a virtual environment (recommended):
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

5. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## 🚀 Usage

1. Prepare your audio data:
   - Create a directory for your audio files
   - Organize audio files into subdirectories, where each subdirectory name is the class label
   ```
   data/
   ├── class1/
   │   ├── audio1.wav
   │   └── audio2.wav
   ├── class2/
   │   ├── audio3.wav
   │   └── audio4.wav
   ```

2. Use the classifier in your Python code:
   ```python
   from sound_classifier import SoundClassifier

   # Initialize classifier
   classifier = SoundClassifier(
       data_dir='path/to/data',
       model_type='rf',  # Options: 'rf', 'lr', 'svm', 'nn'
       sr=22050,
       duration=20
   )

   # Train the model
   classifier.train()

   # Save the model
   classifier.save_model('model.pkl')

   # Load a saved model
   classifier.load_model('model.pkl')

   # Predict a new audio file
   prediction = classifier.predict('path/to/audio.wav')
   ```

## 🔍 How It Works

The Sound Classifier processes audio files through several steps:

1. **Feature Extraction**:
   - Loads audio files using librosa
   - Extracts MFCC (Mel-frequency cepstral coefficients)
   - Computes spectral features (centroid, rolloff)
   - Calculates statistical measures

2. **Data Preprocessing**:
   - Standardizes features using StandardScaler
   - Encodes class labels
   - Splits data into training and testing sets

3. **Model Training**:
   - Supports multiple algorithms:
     - Random Forest (rf)
     - Logistic Regression (lr)
     - Support Vector Machine (svm)
     - Neural Network (nn)
   - Trains on the preprocessed data
   - Evaluates performance using classification metrics

## 🔧 Troubleshooting

### Common Issues:

1. **Installation Errors**:
   - Make sure you have the latest pip version:
     ```
     python -m pip install --upgrade pip
     ```
   - On Windows, if you get build errors, install Visual C++ Build Tools
   - On Mac, if you get librosa errors, install libsndfile:
     ```
     brew install libsndfile
     ```

2. **Import Errors**:
   - Ensure you're in the virtual environment
   - Verify all dependencies are installed:
     ```
     pip list
     ```

3. **Audio File Issues**:
   - Make sure audio files are in WAV format
   - Check if the sample rate matches your settings
   - Verify file permissions

For additional help or to report issues, please create an issue in the repository.
