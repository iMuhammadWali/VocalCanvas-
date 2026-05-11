import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from sklearn.utils.class_weight import (
    compute_class_weight
)

import tensorflow as tf

from tensorflow.keras import (
    layers,
    models,
    Input
)

from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array
)

# =========================================================
# Config
# =========================================================
SPECTROGRAMS_DIR = "spectrograms"

SPEAKERS = [
    "ahmed",
    "zoha"
]

MODEL_SAVE_PATH = (
    "models/vocalcanvas_sanity.keras"
)

os.makedirs("models", exist_ok=True)

IMG_HEIGHT = 128
IMG_WIDTH = 128

# =========================================================
# Load PNG spectrograms
# =========================================================
def load_split(split_name):

    X = []
    y = []

    for label, speaker in enumerate(SPEAKERS):

        folder = os.path.join(
            SPECTROGRAMS_DIR,
            split_name,
            speaker
        )

        files = [

            f for f in os.listdir(folder)
            if f.endswith(".png")

        ]

        for file in files:

            path = os.path.join(
                folder,
                file
            )

            # Load image
            img = load_img(
                path,
                color_mode="grayscale",
                target_size=(
                    IMG_HEIGHT,
                    IMG_WIDTH
                )
            )

            # Convert to numpy
            img_array = img_to_array(img)

            # Normalize
            img_array = img_array / 255.0

            X.append(img_array)
            y.append(label)

    return np.array(X), np.array(y)

# =========================================================
# Load datasets
# =========================================================
print("Loading spectrogram images...")

X_train, y_train = load_split("train")
X_val, y_val = load_split("val")
X_test, y_test = load_split("test")

print(
    f"Train: {len(X_train)} | "
    f"Val: {len(X_val)} | "
    f"Test: {len(X_test)}"
)

print(f"Dataset shape: {X_train.shape}")

# =========================================================
# Class Weights
# =========================================================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = dict(
    enumerate(class_weights)
)

print(
    f"\nClass weights: "
    f"{class_weight_dict}"
)

# =========================================================
# Build CNN
# =========================================================
print("\nBuilding CNN...")

input_shape = X_train.shape[1:]

model = models.Sequential([

    Input(shape=input_shape),

    # =====================================================
    # Block 1
    # =====================================================
    layers.Conv2D(
        32,
        (3, 3),
        activation='relu'
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Dropout(0.3),

    # =====================================================
    # Block 2
    # =====================================================
    layers.Conv2D(
        64,
        (3, 3),
        activation='relu'
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Dropout(0.3),

    # =====================================================
    # Block 3
    # =====================================================
    layers.Conv2D(
        128,
        (3, 3),
        activation='relu'
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Dropout(0.3),

    # =====================================================
    # Dense Layers
    # =====================================================
    layers.Flatten(),

    layers.Dense(
        64,
        activation='relu'
    ),

    layers.Dropout(0.5),

    # =====================================================
    # Output
    # =====================================================
    layers.Dense(
        len(SPEAKERS),
        activation='softmax'
    )

])

model.summary()

# =========================================================
# Compile
# =========================================================
model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)

# =========================================================
# Train
# =========================================================
print("\nTraining...")

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=50,

    batch_size=16,

    class_weight=class_weight_dict,

    callbacks=[

        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        )

    ]

)

# =========================================================
# Evaluate
# =========================================================
print("\nEvaluating on test set...")

test_loss, test_acc = model.evaluate(
    X_test,
    y_test
)

print(
    f"Test Accuracy: "
    f"{test_acc * 100:.2f}%"
)

y_pred = np.argmax(
    model.predict(X_test),
    axis=1
)

print("\nClassification Report:")

print(

    classification_report(
        y_test,
        y_pred,
        target_names=SPEAKERS,
        zero_division=0
    )

)

# =========================================================
# Confusion Matrix
# =========================================================
cm = confusion_matrix(
    y_test,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=SPEAKERS
)

disp.plot(cmap='Blues')

plt.title(
    "VocalCanvas — Confusion Matrix"
)

plt.savefig("confusion_matrix.png")

plt.show()

# =========================================================
# Training Curves
# =========================================================
fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(12, 4)
)

# Accuracy
ax1.plot(
    history.history['accuracy'],
    label='Train'
)

ax1.plot(
    history.history['val_accuracy'],
    label='Validation'
)

ax1.set_title('Accuracy')

ax1.legend()

# Loss
ax2.plot(
    history.history['loss'],
    label='Train'
)

ax2.plot(
    history.history['val_loss'],
    label='Validation'
)

ax2.set_title('Loss')

ax2.legend()

plt.suptitle(
    "VocalCanvas — Training Curves"
)

plt.savefig("training_curves.png")

plt.show()

# =========================================================
# Save Model
# =========================================================
model.save(MODEL_SAVE_PATH)

print(
    f"\nModel saved to "
    f"{MODEL_SAVE_PATH}"
)