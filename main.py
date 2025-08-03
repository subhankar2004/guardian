from flask import Flask, request, jsonify
import librosa
import numpy as np
import joblib
import os
import tensorflow as tf
from keras.models import load_model

app = Flask(__name__)

# Load models
try:
    svm_model = joblib.load("svm_model.pkl")
    mlp = load_model("mlp_model.h5")
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    svm_model = None
    mlp = None

def extract_mfcc(audio_path):
    """Extract MFCC features from audio file"""
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=60)
        mfcc_mean = np.mean(mfcc, axis=1)
        return mfcc_mean.reshape(1, -1)
    except Exception as e:
        print(f"Error extracting MFCC: {e}")
        return None

def get_alert_level(audio_path):
    """Predict alert level using both SVM and MLP models"""
    try:
        features = extract_mfcc(audio_path)
        if features is None:
            return "Error"
        
        # SVM prediction
        svm_pred = svm_model.predict(features)[0]
        
        # MLP prediction
        mlp_pred_probs = mlp.predict(features, verbose=0)
        mlp_pred = np.argmax(mlp_pred_probs, axis=1)[0]

        # Combined decision logic
        if svm_pred == 1 and mlp_pred == 1:
            return "High Alert"
        elif svm_pred == 1 or mlp_pred == 1:
            return "Moderate Alert"
        else:
            return "Normal"
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Error"

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for audio classification"""
    # Check if models are loaded
    if svm_model is None or mlp is None:
        return jsonify({'error': 'Models not loaded properly'}), 500
    
    # Check if audio file is provided
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']
    
    # Check if file has a name
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Save temporary file
        temp_path = "temp.wav"
        file.save(temp_path)

        # Get prediction
        alert_level = get_alert_level(temp_path)
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if alert_level == "Error":
            return jsonify({'error': 'Error processing audio file'}), 500

        return jsonify({'alert_level': alert_level})
    
    except Exception as e:
        # Clean up in case of error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': svm_model is not None and mlp is not None,
        'tensorflow_version': tf.__version__
    })

if __name__ == "__main__":
    print(f"TensorFlow version: {tf.__version__}")
    app.run(host="0.0.0.0", port=5000)