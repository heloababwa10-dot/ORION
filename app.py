from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import zipfile
import tempfile
import logging
import tensorflow as tf
from tensorflow import keras

os.environ['PYTHONUNBUFFERED'] = '1'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[logging.StreamHandler()]
)
from tensorflow.keras import layers
import tensorflow.keras.backend as K

# ==================== CONFIG ====================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

IMG_SIZE = (256, 256)

# ==================== MODEL ARCHITECTURE ====================

def build_fresh_model(num_classes=2):
    """
    Build model with a CLEAN Keras name counter so layer names always start
    at conv2d, conv2d_1 ... matching exactly what is stored in the .keras files.
    Must be called ONCE per model load, with the counter reset beforehand.
    """
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
    Load a Keras 2.x .keras file into the current TF/Keras version.

    The .keras format is a ZIP containing:
      - config.json   (architecture)
      - metadata.json (keras version)
      - model.weights.h5 (actual weights, named by layer name)

    The key fix: reset Keras's global layer name counter before building
    the model so names match exactly what is in model.weights.h5.
    """

    # ── Strategy 1: standard load (works when TF versions match) ──────────
    try:
        model = keras.models.load_model(keras_path)
        logging.info(f"   ✅ Standard load succeeded")
        return model
    except Exception as e:
        logging.info(f"   ⚠️  Standard load failed, using weight extraction...")

    # ── Extract model.weights.h5 from the ZIP ─────────────────────────────
    with zipfile.ZipFile(keras_path, 'r') as z:
        h5_files = [n for n in z.namelist() if n.endswith('.h5')]
        if not h5_files:
            raise RuntimeError(f"No .h5 file inside {keras_path}. Contents: {z.namelist()}")
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp.write(z.read(h5_files[0]))
            tmp_path = tmp.name

    try:
        # ── Strategy 2: reset counter → build → load by name ──────────────
        # Resetting the counter ensures layer names start from conv2d, conv2d_1 ...
        # which exactly matches the names stored in model.weights.h5.
        K.reset_uids()

        model = build_fresh_model(num_classes=2)
        model.build(input_shape=(None, *IMG_SIZE, 3))

        # Print built model layer names
        layer_names = [l.name for l in model.layers]
        logging.info(f"   Built model layers: {layer_names}")

        # Peek inside H5 to see what layer names are stored
        import h5py
        h5_layer_names = []
        with h5py.File(tmp_path, 'r') as hf:
            def collect_keys(name, obj):
                if isinstance(obj, h5py.Group) and 'vars' in obj:
                    h5_layer_names.append(name)
            hf.visititems(collect_keys)
        logging.info(f"   H5 stored layers: {h5_layer_names}")

        missing = [n for n in layer_names if not any(n in h for h in h5_layer_names)]
        if missing:
            logging.info(f"   ⚠️  Layers with no H5 match (weights NOT loaded): {missing}")

        model.load_weights(tmp_path, by_name=True, skip_mismatch=True)

        total_weights = len(model.weights)
        nonzero_weights = sum(1 for w in model.weights if float(tf.reduce_sum(tf.abs(w)).numpy()) > 1e-6)
        logging.info(f"   ✅ Weight extraction (by_name) succeeded")
        logging.info(f"   📊 Weights check: {nonzero_weights}/{total_weights} tensors are non-zero")
        if nonzero_weights < total_weights * 0.5:
            logging.info(f"   🚨 CRITICAL: <50% weights loaded — model is mostly random!")
        return model

    except Exception as e:
        logging.info(f"   ⚠️  by_name load failed ({e}), trying positional load...")

        # ── Strategy 3: reset counter again → positional load ─────────────
        try:
            K.reset_uids()
            model = build_fresh_model(num_classes=2)
            model.build(input_shape=(None, *IMG_SIZE, 3))
            model.load_weights(tmp_path)
            logging.info(f"   ✅ Positional weight load succeeded")
            return model
        except Exception as e2:
            raise RuntimeError(f"All strategies failed. Last error: {e2}")
    finally:
        os.unlink(tmp_path)


# ==================== CLASSIFIER ====================

class IronWaterClassifier:
    def __init__(self, object_type, model_path):
        self.object_type = object_type

        logging.info(f"\n🔄 Loading: {object_type}")
        try:
            self.model = load_keras2_model(model_path)
            logging.info(f"✅ Ready: {object_type}")
        except Exception as e:
            logging.info(f"❌ Failed: {object_type} — {e}")
            self.model = None

        if object_type == 'orange':
            self.class_names = ['orange_clean', 'orange_iron_contaminated']
        elif object_type == 'banana':
            self.class_names = ['banana_clean', 'banana_iron_contaminated']
        elif object_type == 'egg':
            self.class_names = ['egg_clean', 'egg_iron_contaminated']

        self.condition_map = {
            'clean':              'Clean Water',
            'iron_contaminated':  'Iron Contaminated Water'
        }

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))

        # CLAHE contrast enhancement (same as training)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # The model already has Rescaling(1/255) as its first layer,
        # so we feed raw uint8 values [0, 255] — do NOT divide here.
        return np.array(img, dtype=np.float32)

    def classify(self, img):
        if self.model is None:
            raise ValueError(f"Model '{self.object_type}' not loaded")

        preprocessed = self.preprocess(img)
        batch = np.expand_dims(preprocessed, axis=0)

        pred = self.model.predict(batch, verbose=0)[0]
        logging.info(f"   Raw predictions [{self.object_type}]: {pred}")

        idx = int(np.argmax(pred))
        confidence = float(pred[idx])

        class_name = self.class_names[idx]
        condition_key = "_".join(class_name.split('_')[1:])
        condition = self.condition_map.get(condition_key, condition_key)

        return {
            "condition":  condition,
            "confidence": confidence,
            "raw_pred":   [float(p) for p in pred]  # debug — remove later
        }


# ==================== LOAD MODELS ====================

classifiers = {}

def load_models():
    model_files = {
        'orange': os.path.join(MODELS_FOLDER, 'orange_classifier.keras'),
        'banana': os.path.join(MODELS_FOLDER, 'banana_classifier.keras'),
        'egg':    os.path.join(MODELS_FOLDER, 'egg_classifier.keras'),
    }

    for obj, path in model_files.items():
        if os.path.exists(path):
            classifiers[obj] = IronWaterClassifier(obj, path)
        else:
            logging.info(f"⚠️  Not found: {path}")

    loaded = [k for k, v in classifiers.items() if v.model is not None]
    failed = [k for k, v in classifiers.items() if v.model is None]
    logging.info(f"\n{'='*50}")
    logging.info(f"✅ Loaded: {loaded}")
    if failed:
        logging.info(f"❌ Failed: {failed}")
    logging.info(f"{'='*50}\n")


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
        object_type = request.form.get('object_type', '').strip()

        if object_type not in classifiers:
            return jsonify({'error': f"Unknown model '{object_type}'. Available: {list(classifiers.keys())}"}), 400

        if classifiers[object_type].model is None:
            return jsonify({'error': f"Model '{object_type}' failed to load at startup"}), 503

        # Accept image from any field name (handles camera blob edge cases)
        file = request.files.get('image') or (list(request.files.values())[0] if request.files else None)
        if file is None:
            return jsonify({'error': 'No image in request'}), 400

        img_bytes = file.read()
        if len(img_bytes) == 0:
            return jsonify({'error': 'Empty file received'}), 400

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Could not decode image — unsupported format?'}), 400

        result = classifiers[object_type].classify(img)
        return jsonify({'success': True, 'result': result})

    except Exception as e:
        import traceback
        logging.info("❌ ERROR: %s", traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# ==================== MAIN ====================

if __name__ == '__main__':
    logging.info("=" * 50)
    logging.info("🚀 STARTING ORION")
    logging.info("=" * 50)

    load_models()

    logging.info("🌐 Open: http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)
