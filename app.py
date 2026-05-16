from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
from tensorflow import keras

# ==================== CONFIG ====================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==================== CLASSIFIER ====================

class IronWaterClassifier:
    def __init__(self, object_type, model_path):
        self.object_type = object_type

        try:
            self.model = keras.models.load_model(model_path)
            print(f"✅ Model loaded: {object_type}")
        except Exception as e:
            print(f"❌ Failed load {object_type}: {e}")
            self.model = None

        if object_type == 'orange':
            self.class_names = ['orange_clean', 'orange_iron_contaminated']
        elif object_type == 'banana':
            self.class_names = ['banana_clean', 'banana_iron_contaminated']
        elif object_type == 'egg':
            self.class_names = ['egg_clean', 'egg_iron_contaminated']

        self.condition_map = {
            'clean': 'Clean Water',
            'iron_contaminated': 'Iron Contaminated Water'
        }

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))

        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(2.0, (8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return np.array(img, dtype=np.float32) / 255.0

    def classify(self, img):
        if self.model is None:
            raise ValueError("Model not loaded")

        img = self.preprocess(img)
        img = np.expand_dims(img, axis=0)

        pred = self.model.predict(img, verbose=0)[0]
        idx = np.argmax(pred)
        confidence = float(pred[idx])

        class_name = self.class_names[idx]
        condition_key = "_".join(class_name.split('_')[1:])
        condition = self.condition_map.get(condition_key, condition_key)

        return {
            "condition": condition,
            "confidence": confidence
        }

# ==================== LOAD MODELS ====================

classifiers = {}

def load_models():
    model_files = {
        'orange': os.path.join(MODELS_FOLDER, 'orange_classifier.keras'),
        'banana': os.path.join(MODELS_FOLDER, 'banana_classifier.keras'),
        'egg': os.path.join(MODELS_FOLDER, 'egg_classifier.keras')
    }

    for obj, path in model_files.items():
        if os.path.exists(path):
            classifiers[obj] = IronWaterClassifier(obj, path)
        else:
            print(f"⚠️ Model not found: {path}")

    print("Loaded models:", list(classifiers.keys()))

# ==================== ROUTES ====================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload')
def upload_page():
    return render_template('upload.html')


@app.route('/api/classify-image', methods=['POST'])
def classify_image():
    try:
        object_type = request.form.get('object_type')
        print("Object:", object_type)

        if object_type not in classifiers:
            return jsonify({'error': 'Invalid object type'}), 400

        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        img_bytes = file.read()

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400

        result = classifiers[object_type].classify(img)

        return jsonify({
            'success': True,
            'result': result
        })  

    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({'error': str(e)}), 500


# ==================== MAIN ====================

if __name__ == '__main__':
    print("="*50)
    print("🚀 STARTING APP")
    print("="*50)

    load_models()

    print("🌐 Open: http://localhost:5000/upload")
    app.run(debug=True)
