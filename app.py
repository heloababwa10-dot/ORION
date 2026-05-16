from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import zipfile
import tempfile
from tensorflow import keras
from tensorflow.keras import layers

# ==================== CONFIG ====================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==================== MODEL ARCHITECTURE ====================
# Must exactly match the architecture used during training.

IMG_SIZE = (256, 256)

def build_model(num_classes=2):
    model = keras.Sequential([
        layers.InputLayer(input_shape=(*IMG_SIZE, 3)),
        layers.Rescaling(1./255),

        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 4
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.GlobalAveragePooling2D(),

        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def load_keras2_model(keras_path):
    """
    Load a Keras 2.x .keras file (ZIP containing model.weights.h5) into a
    freshly built Keras 3 model by extracting and loading just the weights.
    """
    # Strategy 1: standard load (works if versions match)
    try:
        model = keras.models.load_model(keras_path)
        print(f"   ✅ Standard load succeeded")
        return model
    except Exception as e:
        print(f"   ⚠️  Standard load failed ({e}), trying weight extraction...")

    # Strategy 2: extract model.weights.h5 from the ZIP and load into fresh model
    try:
        with zipfile.ZipFile(keras_path, 'r') as z:
            names = z.namelist()
            weight_files = [n for n in names if n.endswith('.h5')]
            if not weight_files:
                raise ValueError(f"No .h5 weight file found inside {keras_path}. Contents: {names}")

            weight_entry = weight_files[0]
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                tmp.write(z.read(weight_entry))
                tmp_path = tmp.name

        model = build_model(num_classes=2)
        model.build(input_shape=(None, *IMG_SIZE, 3))
        model.load_weights(tmp_path)
        os.unlink(tmp_path)
        print(f"   ✅ Weight extraction load succeeded")
        return model

    except Exception as e:
        print(f"   ⚠️  Weight extraction failed ({e}), trying by_name=True...")

    # Strategy 3: same but load weights by layer name (more lenient)
    try:
        with zipfile.ZipFile(keras_path, 'r') as z:
            weight_files = [n for n in z.namelist() if n.endswith('.h5')]
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                tmp.write(z.read(weight_files[0]))
                tmp_path = tmp.name

        model = build_model(num_classes=2)
        model.build(input_shape=(None, *IMG_SIZE, 3))
        model.load_weights(tmp_path, by_name=True, skip_mismatch=True)
        os.unlink(tmp_path)
        print(f"   ✅ by_name weight load succeeded")
        return model

    except Exception as e:
        raise RuntimeError(f"All loading strategies failed: {e}")


# ==================== CLASSIFIER ====================

class IronWaterClassifier:
    def __init__(self, object_type, model_path):
        self.object_type = object_type

        print(f"🔄 Loading model: {object_type}")
        try:
            self.model = load_keras2_model(model_path)
            print(f"✅ Model ready: {object_type}")
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

        # NOTE: Model has Rescaling(1/255) built in — do NOT divide by 255 here.
        return np.array(img, dtype=np.float32)

    def classify(self, img):
        if self.model is None:
            raise ValueError(f"Model '{self.object_type}' failed to load at startup")

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
        'egg':    os.path.join(MODELS_FOLDER, 'egg_classifier.keras')
    }

    for obj, path in model_files.items():
        if os.path.exists(path):
            classifiers[obj] = IronWaterClassifier(obj, path)
        else:
            print(f"⚠️  Model file not found: {path}")

    loaded = [k for k, v in classifiers.items() if v.model is not None]
    failed = [k for k, v in classifiers.items() if v.model is None]
    print(f"✅ Models ready: {loaded}")
    if failed:
        print(f"❌ Models failed: {failed}")

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
            return jsonify({'error': f"Invalid object type '{object_type}'. Valid: {list(classifiers.keys())}"}), 400

        if classifiers[object_type].model is None:
            return jsonify({'error': f"Model '{object_type}' failed to load at startup"}), 503

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
    print("=" * 50)
    print("🚀 STARTING APP")
    print("=" * 50)

    load_models()

    print("🌐 Open: http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)
