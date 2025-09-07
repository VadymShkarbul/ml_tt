import os
from pathlib import Path

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from model import build_model

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data' / 'split'
TRAIN_DIR = DATA_DIR / 'train'
VAL_DIR = DATA_DIR / 'val'
MODELS_DIR = ROOT / 'models'
SAVE_PATH = MODELS_DIR / 'screen_detector.h5'

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20


def main():
    assert TRAIN_DIR.exists(), f"Train dir not found: {TRAIN_DIR}. Run data_preparation.py first."
    assert VAL_DIR.exists(), f"Val dir not found: {VAL_DIR}. Run data_preparation.py first."

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Data generators
    train_gen = ImageDataGenerator(
        rescale=1./255.0,
        horizontal_flip=True,
        rotation_range=15,
    )
    val_gen = ImageDataGenerator(
        rescale=1./255.0,
    )

    train_flow = train_gen.flow_from_directory(
        str(TRAIN_DIR),
        target_size=IMG_SIZE,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )

    val_flow = val_gen.flow_from_directory(
        str(VAL_DIR),
        target_size=IMG_SIZE,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    # Build model
    model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), learning_rate=1e-3)

    # Callbacks
    callbacks = [
        EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
        ModelCheckpoint(filepath=str(SAVE_PATH), save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(patience=3, factor=0.5, monitor='val_loss')
    ]

    # Fit
    model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Ensure save (in case the best checkpoint equals the last epoch)
    if not SAVE_PATH.exists():
        model.save(str(SAVE_PATH))
    print(f"Model saved to {SAVE_PATH}")


if __name__ == '__main__':
    main()
