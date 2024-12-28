import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import joblib


class SoundClassifier:
    def __init__(self, data_dir, model_type='rf', sr=22050, duration=20):
        self.data_dir = data_dir
        self.sr = sr
        self.duration = duration
        self.model = None
        self.le = LabelEncoder()
        self.scaler = StandardScaler()
        self.model_type = model_type

    def extract_features(self, file_path):
        # Load audio file
        y, _ = librosa.load(file_path, sr=self.sr, duration=self.duration)

        # Pad or truncate to fixed length
        if len(y) < self.sr * self.duration:
            y = np.pad(y, (0, self.sr * self.duration - len(y)))
        else:
            y = y[:self.sr * self.duration]

        # Extract features
        mfccs = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr)

        # Compute statistics
        features = np.concatenate([
            mfccs.mean(axis=1),
            mfccs.std(axis=1),
            spectral_centroid.mean(axis=1),
            spectral_rolloff.mean(axis=1)
        ])

        return features

    def prepare_data(self):
        X = []
        y = []

        # Iterate through each issue folder
        for issue in os.listdir(self.data_dir):
            issue_path = os.path.join(self.data_dir, issue)
            if os.path.isdir(issue_path):
                # Process each audio file in the folder
                for audio_file in os.listdir(issue_path):
                    if audio_file.endswith('.wav'):
                        file_path = os.path.join(issue_path, audio_file)
                        features = self.extract_features(file_path)
                        X.append(features)
                        y.append(issue)
        print(len(X))
        print(len(y))
        X = np.array(X)
        y = self.le.fit_transform(y)

        return X, y

    def train(self):
        # Prepare data
        X, y = self.prepare_data()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model based on model_type
        if self.model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == 'lr':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif self.model_type == 'svm':
            self.model = SVC(kernel='rbf', random_state=42)
        elif self.model_type == 'nn':
            self.model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        else:
            raise ValueError("Invalid model type. Choose 'rf', 'lr', 'svm', or 'nn'.")

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        print(f"\nModel Performance ({self.model_type}):")
        print(classification_report(y_test, y_pred,
                                    labels=np.unique(y),
                                    target_names=self.le.classes_[np.unique(y)]))

        return self.model

    def predict(self, audio_file):
        # Extract features from new audio
        features = self.extract_features(audio_file)

        # Scale features
        features_scaled = self.scaler.transform([features])

        # Make prediction
        prediction = self.model.predict(features_scaled)[0]

        # Return the issue name
        return self.le.inverse_transform([prediction])[0]

    def save_model(self, model_path='sound_classifier_model.joblib'):
        """Save the trained model, label encoder, and scaler"""
        if self.model is None:
            raise ValueError("Model hasn't been trained yet!")

        model_data = {
            'model': self.model,
            'label_encoder': self.le,
            'scaler': self.scaler,
            'model_type': self.model_type
        }
        joblib.dump(model_data, model_path)

    @classmethod
    def load_model(cls, model_path='sound_classifier_model.joblib'):
        """Load a trained model"""
        classifier = cls(data_dir=None)  # Create instance without data dir
        model_data = joblib.load(model_path)
        classifier.model = model_data['model']
        classifier.le = model_data['label_encoder']
        classifier.scaler = model_data['scaler']
        classifier.model_type = model_data['model_type']
        return classifier