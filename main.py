#!/usr/bin/env python3
import os
import sys
import warnings
import gc

# Memory optimizations BEFORE any imports
os.environ.update({
    'TF_CPP_MIN_LOG_LEVEL': '3',
    'CUDA_VISIBLE_DEVICES': '-1',
    'NUMBA_DISABLE_JIT': '1',
    'NUMBA_CACHE_DIR': '/tmp',
    'LIBROSA_CACHE_DIR': '/tmp',
    'OMP_NUM_THREADS': '1',
    'PYTHONDONTWRITEBYTECODE': '1'
})

warnings.filterwarnings('ignore')

# Core imports
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import logging
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime

# Configure minimal logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB max
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a'}

# Global variables
svm_model = None
mlp = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    """Load models with memory optimization"""
    global svm_model, mlp
    
    print("🔄 Loading models...")
    
    # Load SVM first (smaller)
    try:
        if os.path.exists("svm_model.pkl"):
            svm_model = joblib.load("svm_model.pkl")
            print("✅ SVM model loaded")
        else:
            print("❌ svm_model.pkl not found")
    except Exception as e:
        print(f"❌ SVM loading failed: {e}")
        svm_model = None

    # Force garbage collection between models
    gc.collect()

    # Load MLP with lazy TensorFlow import
    try:
        if os.path.exists("mlp_model.h5"):
            print("Loading TensorFlow...")
            
            # Try different import methods
            try:
                from keras.models import load_model
                print("Using standalone Keras")
            except ImportError:
                import tensorflow as tf
                # Memory optimization for TensorFlow
                tf.config.threading.set_intra_op_parallelism_threads(1)
                tf.config.threading.set_inter_op_parallelism_threads(1)
                from tensorflow.keras.models import load_model
                print("Using TensorFlow Keras")
            
            mlp = load_model("mlp_model.h5", compile=False)
            print("✅ MLP model loaded")
        else:
            print("❌ mlp_model.h5 not found")
    except Exception as e:
        print(f"❌ MLP loading failed: {e}")
        mlp = None

    # Final cleanup
    gc.collect()
    
    models_loaded = sum([svm_model is not None, mlp is not None])
    print(f"📊 Models loaded: {models_loaded}/2")
    return models_loaded > 0

def extract_mfcc_optimized(audio_path):
    """Memory-optimized MFCC extraction"""
    try:
        # Lazy import librosa
        import librosa
        librosa.set_fftlib('numpy')  # Use numpy FFT
        
        # Load with strict memory limits
        audio, sr = librosa.load(
            audio_path, 
            sr=16000,        # Low sample rate
            duration=12,     # Max 12 seconds
            mono=True,
            res_type='kaiser_fast'
        )
        
        # Extract minimal MFCCs
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=sr, 
            n_mfcc=30,       # Reduced features
            n_fft=512,       # Small window
            hop_length=256
        )
        
        # Get statistics
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Combine features
        features = np.concatenate([mfcc_mean, mfcc_std])
        
        # Cleanup
        del audio, mfcc, mfcc_mean, mfcc_std
        gc.collect()
        
        # Ensure correct size for models (pad/truncate to 60)
        if len(features) < 60:
            features = np.pad(features, (0, 60 - len(features)))
        else:
            features = features[:60]
            
        return features.reshape(1, -1)
        
    except Exception as e:
        print(f"MFCC extraction error: {e}")
        return None

def get_alert_level(audio_path):
    """Optimized prediction function"""
    try:
        features = extract_mfcc_optimized(audio_path)
        if features is None:
            return "Error", None
        
        # SVM prediction
        svm_pred = 0
        svm_confidence = 0.0
        if svm_model:
            svm_pred = svm_model.predict(features)[0]
            if hasattr(svm_model, 'predict_proba'):
                try:
                    svm_proba = svm_model.predict_proba(features)[0]
                    svm_confidence = max(svm_proba)
                except:
                    svm_confidence = 0.5
        
        # MLP prediction
        mlp_pred = 0
        mlp_confidence = 0.0
        if mlp:
            try:
                mlp_pred_probs = mlp.predict(features, verbose=0)
                mlp_pred = np.argmax(mlp_pred_probs, axis=1)[0]
                mlp_confidence = np.max(mlp_pred_probs)
                del mlp_pred_probs
            except Exception as e:
                print(f"MLP prediction error: {e}")
                mlp_pred = 0

        # Cleanup
        del features
        gc.collect()

        # Decision logic
        if svm_pred == 1 and mlp_pred == 1:
            alert_level = "High Alert"
        elif svm_pred == 1 or mlp_pred == 1:
            alert_level = "Moderate Alert"
        else:
            alert_level = "Normal"
            
        details = {
            'svm_prediction': int(svm_pred),
            'mlp_prediction': int(mlp_pred),
            'svm_confidence': float(svm_confidence),
            'mlp_confidence': float(mlp_confidence)
        }
        
        return alert_level, details
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error", None

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'name': '🎵 Audio Alert Classification API',
        'version': '1.0.0',
        'status': 'healthy',
        'description': 'Memory-optimized AI audio classification',
        'models_loaded': {
            'svm': svm_model is not None,
            'mlp': mlp is not None
        },
        'endpoints': {
            'POST /predict': 'Upload audio for classification',
            'GET /health': 'Check API health'
        },
        'limits': {
            'max_file_size': '8MB',
            'supported_formats': list(ALLOWED_EXTENSIONS)
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Audio classification endpoint"""
    if svm_model is None and mlp is None:
        return jsonify({
            'error': 'No models available',
            'details': 'Both SVM and MLP models failed to load'
        }), 500
    
    if 'audio' not in request.files:
        return jsonify({
            'error': 'No audio file provided',
            'hint': 'Send audio file with key "audio"'
        }), 400

    file = request.files['audio']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            'error': 'Unsupported file format',
            'supported_formats': list(ALLOWED_EXTENSIONS)
        }), 400

    temp_path = None
    try:
        # Create secure temp file
        filename = secure_filename(file.filename)
        temp_path = f"/tmp/audio_{uuid.uuid4().hex[:8]}.wav"
        file.save(temp_path)

        print(f"Processing: {filename}")

        # Get prediction
        alert_level, details = get_alert_level(temp_path)

        if alert_level == "Error":
            return jsonify({
                'error': 'Audio processing failed',
                'hint': 'Check if file is valid audio format'
            }), 500

        response = {
            'alert_level': alert_level,
            'timestamp': datetime.utcnow().isoformat(),
            'filename': filename,
            'models_used': {
                'svm': svm_model is not None,
                'mlp': mlp is not None
            }
        }
        
        if details:
            response['prediction_details'] = details

        return jsonify(response)
    
    except Exception as e:
        print(f"Request error: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500
    
    finally:
        # Cleanup
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        gc.collect()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        models_loaded = sum([svm_model is not None, mlp is not None])
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'models': {
                'svm_loaded': svm_model is not None,
                'mlp_loaded': mlp is not None,
                'total_loaded': models_loaded
            },
            'memory_optimized': True
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'error': 'File too large',
        'max_size': '8MB'
    }), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'error': 'Internal server error'
    }), 500

if __name__ == "__main__":
    print("🚀 Starting Memory-Optimized Audio Alert API...")
    
    # Load models on startup
    if load_models():
        print("✅ Ready to serve requests!")
    else:
        print("⚠️ No models loaded - limited functionality")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port, debug=False)