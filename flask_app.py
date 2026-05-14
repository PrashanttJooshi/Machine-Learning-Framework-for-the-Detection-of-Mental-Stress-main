from flask import Flask, render_template, Response, jsonify, request
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import os
import sys
import traceback

app = Flask(__name__, template_folder='Template')
# app = Flask(__name__)

# Global variables
stress_levels = []
emotions = []
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise', 'Stress']
emotion_counts = {label: 0 for label in emotion_labels}

print("=" * 80)
print("FLASK APP STARTING...")
print("=" * 80)

# Try to import heavy dependencies
try:
    import cv2
    print("✅ OpenCV imported successfully")
except Exception as e:
    print(f"❌ OpenCV import failed: {e}")
    cv2 = None

try:
    import numpy as np
    print("✅ NumPy imported successfully")
except Exception as e:
    print(f"❌ NumPy import failed: {e}")
    np = None

try:
    from keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
    print("✅ Keras/TensorFlow imported successfully")
except Exception as e:
    print(f"❌ Keras/TensorFlow import failed: {e}")
    load_model = None

# Load face cascade
face_cascade = None
try:
    if cv2:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print("✅ Face cascade loaded successfully")
except Exception as e:
    print(f"❌ Face cascade failed: {e}")

# Model loading with error handling
classifier = None

def load_emotion_model():
    """Load the emotion classifier model"""
    global classifier
    if classifier is None:
        try:
            if load_model is None:
                print("⚠️  Keras not available, cannot load model")
                return None
            
            classifier = load_model('model_weights_78.h5')
            print("✅ Emotion model loaded successfully")
        except FileNotFoundError:
            print("❌ ERROR: model_weights_78.h5 not found in /app/")
            print(f"Current directory: {os.getcwd()}")
            print(f"Files in current directory: {os.listdir('.')}")
            return None
        except Exception as e:
            print(f"❌ Model loading error: {e}")
            traceback.print_exc()
            return None
    return classifier

print("\nAttempting to load model...")
load_emotion_model()
print("=" * 80)

@app.route('/')
def index():
    """Home page"""
    try:
        print("Loading index.html...")
        return render_template('index.html')
    except Exception as e:
        print(f"❌ Error in index route: {e}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500

@app.route('/detection')
def detection():
    """Detection page"""
    try:
        return render_template('detection.html')
    except Exception as e:
        print(f"❌ Error in detection route: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard')
def dashboard():
    """Dashboard page"""
    try:
        return render_template('dashboard.html')
    except Exception as e:
        print(f"❌ Error in dashboard route: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/video_feed')
def video_feed():
    """Video feed endpoint"""
    try:
        if cv2 is None:
            return "OpenCV not available", 500
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"❌ Error in video_feed: {e}")
        return jsonify({'error': str(e)}), 500

def generate_frames():
    """Generate video frames"""
    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("⚠️  Camera not available (expected on cloud)")
            yield b''
            return
        
        while True:
            success, frame = camera.read()
            if not success:
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    except Exception as e:
        print(f"❌ Error in generate_frames: {e}")
        yield b''

@app.route('/api/stats')
def get_stats():
    """API endpoint for statistics"""
    try:
        if emotions:
            most_common_emotion = max(set(emotions), key=emotions.count)
            avg_stress = sum(stress_levels) / len(stress_levels) if stress_levels else 0
        else:
            most_common_emotion = "None"
            avg_stress = 0
        
        return jsonify({
            'total_frames': len(emotions),
            'most_common_emotion': most_common_emotion,
            'average_stress': round(avg_stress, 2),
            'emotion_counts': emotion_counts
        })
    except Exception as e:
        print(f"❌ Error in stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'opencv': cv2 is not None,
        'keras': load_model is not None,
        'model_loaded': classifier is not None
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found', 'message': str(error)}), 404

@app.errorhandler(500)
def server_error(error):
    print(f"❌ 500 Error: {error}")
    traceback.print_exc()
    return jsonify({'error': 'Internal server error', 'message': str(error)}), 500

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("🚀 STARTING FLASK APP")
    print("=" * 80)
    
    port = int(os.environ.get('PORT', 10000))
    debug = os.environ.get('DEBUG', 'False') == 'True'
    
    print(f"Port: {port}")
    print(f"Debug: {debug}")
    print("=" * 80 + "\n")
    
    app.run(
        debug=debug,
        host='0.0.0.0',
        port=port,
        threaded=True
    )
