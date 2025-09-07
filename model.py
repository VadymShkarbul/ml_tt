from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC


def build_model(input_shape=(224, 224, 3), learning_rate=1e-3) -> Model:
    base = MobileNetV2(include_top=False, input_shape=input_shape, weights='imagenet')
    base.trainable = False  # freeze

    inp = Input(shape=input_shape)
    x = base(inp, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=out)

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy', AUC(name='auc')])
    return model


if __name__ == '__main__':
    m = build_model()
    m.summary()
