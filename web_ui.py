import gradio as gr
import os
import joblib
from sound_classifier import SoundClassifier
import numpy as np

# Get list of available models and their friendly names
MODELS_DIR = 'models'
MODEL_NAMES = {
    'lr_sound_classifier_model.joblib': 'Logistic Regression',
    'nn_sound_classifier_model.joblib': 'Neural Network',
    'rf_sound_classifier_model.joblib': 'Random Forest',
    'svm_sound_classifier_model.joblib': 'Support Vector Machine'
}

model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('_model.joblib')]
model_choices = {MODEL_NAMES[file]: file for file in model_files}

def load_model(model_file):
    """Load a saved model and its associated scaler and label encoder"""
    model_path = os.path.join(MODELS_DIR, model_file)
    saved_data = joblib.load(model_path)
    return saved_data['model'], saved_data['scaler'], saved_data['label_encoder']

def format_issue(issue_text):
    """Format the issue text to be more readable"""
    # Replace underscores with spaces and title case the text
    formatted = issue_text.replace('_', ' ').title()
    return formatted

def predict_sound(audio_file, model_name):
    """
    Function to make predictions on uploaded audio files using the selected model
    """
    # Get the actual model filename from the friendly name
    model_file = model_choices[model_name]
    
    # Load the selected model
    model, scaler, le = load_model(model_file)
    
    # Initialize classifier for feature extraction only
    classifier = SoundClassifier(data_dir='data')
    
    # Extract features and predict
    features = classifier.extract_features(audio_file)
    features = features.reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    
    # Get the predicted label and format it
    predicted_label = le.inverse_transform(prediction)[0]
    formatted_label = format_issue(predicted_label)
    
    return f"Predicted Issue: {formatted_label}"

# Create Gradio interface
iface = gr.Interface(
    fn=predict_sound,
    inputs=[
        gr.Audio(type="filepath", label="Upload Sound File"),
        gr.Dropdown(choices=list(model_choices.keys()), label="Select Model Type", value=list(model_choices.keys())[0])
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Engine Sound Issue Classifier",
    description="Upload an audio file of engine sound to identify potential issues. Choose from different machine learning models.",
    examples=[
        [os.path.join("test_data", "air_filter_sample_5.wav"), list(model_choices.keys())[0]],
        [os.path.join("test_data", "cd_sample_16.wav"), list(model_choices.keys())[1]],
        [os.path.join("test_data", "vl_sample_4.wav"), list(model_choices.keys())[2]]
    ]
)

if __name__ == "__main__":
    iface.launch()
