from pathlib import Path
from typing import Tuple, IO

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Expect the model under models/screen_detector.h5
MODEL_PATH = Path(__file__).resolve().parent / 'models' / 'screen_detector.h5'

# Lazy-loaded singleton model
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train the model first.")
        _model = load_model(str(MODEL_PATH))
    return _model


def preprocess_image(file_like: IO[bytes], target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Read an image from a file-like object and convert to model-ready numpy batch"""
    image = Image.open(file_like).convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def predict_probability(file_like: IO[bytes]) -> float:
    """Run the model and return the probability (float)"""
    model = get_model()
    batch = preprocess_image(file_like)
    probability = float(model.predict(batch, verbose=0)[0][0])
    return probability
