import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib

from collections import Counter

from sklearn.cluster import KMeans
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    silhouette_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from audio_processor import UnifiedAudioProcessor

# =========================================================
# Config
# =========================================================
PROC = UnifiedAudioProcessor()

SPECTROGRAMS_DIR = "spectrograms"
TEMP_WAV_DIR     = "temp_wavs"

SPEAKERS = [
    "abi",
    "ahmed",
    "zoha"
]

CNN_MODEL_PATH  = "models/vocalcanvas_cnn.keras"
KMEANS_PKL_PATH = "models/vocalcanvas_kmeans.pkl"

os.makedirs("models", exist_ok=True)

IMG_HEIGHT = PROC.IMG_SIZE
IMG_WIDTH  = PROC.IMG_SIZE

# =========================================================
# CNN data loader — loads pre-saved PNG spectrograms
# =========================================================
def load_split(split_name):
    """
    Returns
    -------
    X_img : np.ndarray  (N, 128, 128, 1) float32 [0,1]
    y     : np.ndarray  (N,)             int
    """
    X_img = []
    y     = []

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

            path = os.path.join(folder, file)

            img = load_img(
                path,
                color_mode="grayscale",
                target_size=(IMG_HEIGHT, IMG_WIDTH)
            )

            img_array = img_to_array(img) / 255.0   # (128, 128, 1)

            X_img.append(img_array)
            y.append(label)

    return np.array(X_img), np.array(y)


# =========================================================
# K-Means data loader — loads raw WAV audio from temp_wavs/
#
# Why not reuse the PNG spectrograms?
# MFCCs, spectral contrast, deltas, and ZCR require the
# actual time-domain signal — a resized PNG has already
# lost that information.  We load from temp_wavs/ (original,
# non-augmented audio) to get full acoustic fidelity.
#
# Why no augmentation for K-Means?
# Pitch-shift and time-stretch change the spectral centroid
# and MFCC values, pulling augmented samples away from the
# true cluster centroid.  Augmentation helps CNNs learn
# invariance; it actively hurts distance-based clustering.
# =========================================================
def load_kmeans_features(speakers, temp_wav_dir=TEMP_WAV_DIR):
    """
    Walk temp_wavs/<speaker>/ for each speaker, load each WAV,
    split into 3-second chunks, and extract 144-dim DSP features.

    Returns
    -------
    X_feat : np.ndarray  (N, 144) float32
    y      : np.ndarray  (N,)     int
    """
    X_feat = []
    y      = []

    for label, speaker in enumerate(speakers):

        speaker_dir = os.path.join(temp_wav_dir, speaker)

        if not os.path.isdir(speaker_dir):
            print(
                f"  WARNING: {speaker_dir} not found. "
                f"Run preprocess.py first to populate temp_wavs/."
            )
            continue

        wav_files = sorted([
            f for f in os.listdir(speaker_dir)
            if f.endswith(".wav")
        ])

        if not wav_files:
            print(f"  WARNING: No WAV files in {speaker_dir}")
            continue

        for wav_file in wav_files:

            wav_path = os.path.join(speaker_dir, wav_file)

            audio, _ = librosa.load(wav_path, sr=PROC.SAMPLE_RATE)

            chunks = PROC.split_audio(audio)
            valid  = 0

            for chunk in chunks:

                if PROC.is_silent(chunk):
                    continue

                feat = PROC.chunk_to_kmeans_features(chunk)
                X_feat.append(feat)
                y.append(label)
                valid += 1

            print(
                f"  {speaker}/{wav_file}: "
                f"{valid} valid chunks"
            )

    return np.array(X_feat, dtype=np.float32), np.array(y, dtype=int)


# =========================================================
# Elbow + Silhouette analysis
# =========================================================
def analyze_k_selection(X_scaled, speakers, output_path="kmeans_k_analysis.png"):
    """
    Fit K-Means for k = 2 … max_k and plot inertia (elbow)
    and silhouette score side-by-side.

    Helps verify that k = len(speakers) is actually the most
    natural grouping in the scaled feature space.
    """
    n_samples = len(X_scaled)
    max_k = min(8, n_samples // 5)      # never more than N/5 clusters

    if max_k < 2:
        print("  Not enough samples for k-selection analysis.")
        return

    k_values    = list(range(2, max_k + 1))
    inertias    = []
    sil_scores  = []

    print(f"\n  Testing k = 2 … {max_k}:")

    for k in k_values:

        km = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=42
        )
        km.fit(X_scaled)

        inertia = km.inertia_
        sil     = silhouette_score(X_scaled, km.labels_)

        inertias.append(inertia)
        sil_scores.append(sil)

        marker = " ◄ current" if k == len(speakers) else ""
        print(
            f"    k={k}: inertia={inertia:,.1f}  "
            f"silhouette={sil:.4f}{marker}"
        )

    # --- plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.plot(k_values, inertias, "bo-", linewidth=2, markersize=7)
    ax1.axvline(
        x=len(speakers),
        color="red",
        linestyle="--",
        label=f"k={len(speakers)} (speakers)"
    )
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia  (WCSS)")
    ax1.set_title("Elbow Method")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(k_values, sil_scores, "gs-", linewidth=2, markersize=7)
    ax2.axvline(
        x=len(speakers),
        color="red",
        linestyle="--",
        label=f"k={len(speakers)} (speakers)"
    )
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score  (higher = better)")
    ax2.set_title("Silhouette Score")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle("VocalCanvas — K-Means Optimal-k Analysis")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()

    best_sil_k = k_values[int(np.argmax(sil_scores))]

    print(
        f"\n  Highest silhouette score at k={best_sil_k} "
        f"(score={max(sil_scores):.4f})."
    )

    if best_sil_k != len(speakers):
        print(
            f"  ⚠  NOTE: k={best_sil_k} is the most natural "
            f"grouping mathematically.  Using k={len(speakers)} "
            f"(one per speaker) as requested."
        )

    print(f"  Analysis saved → {output_path}")


# =========================================================
# ███████╗███████╗ ██████╗████████╗██╗ ██████╗ ███╗   ██╗
# SECTION 1 — CNN  (Supervised)
# =========================================================
print("\n" + "=" * 60)
print("SECTION 1 — CNN  (Supervised)")
print("=" * 60)

print("\nLoading spectrogram images...")

X_train_img, y_train = load_split("train")
X_val_img,   y_val   = load_split("val")
X_test_img,  y_test  = load_split("test")

print(
    f"Train: {len(X_train_img)} | "
    f"Val: {len(X_val_img)} | "
    f"Test: {len(X_test_img)}"
)

# --- Class weights ---
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weight_dict}")

# --- Build CNN ---
print("\nBuilding CNN...")

cnn_model = models.Sequential([

    Input(shape=X_train_img.shape[1:]),

    layers.Conv2D(32,  (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(64,  (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(len(SPEAKERS), activation="softmax"),
])

cnn_model.summary()

cnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# --- Train ---
print("\nTraining CNN...")

history = cnn_model.fit(
    X_train_img, y_train,
    validation_data=(X_val_img, y_val),
    epochs=50,
    batch_size=16,
    class_weight=class_weight_dict,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]
)

# --- Evaluate ---
print("\nEvaluating CNN on test set...")

_, test_acc = cnn_model.evaluate(X_test_img, y_test, verbose=0)
cnn_y_pred  = np.argmax(cnn_model.predict(X_test_img, verbose=0), axis=1)

print(f"CNN Test Accuracy: {test_acc * 100:.2f}%")
print("\nCNN Classification Report:")
print(
    classification_report(
        y_test, cnn_y_pred,
        target_names=SPEAKERS,
        zero_division=0
    )
)

# --- Confusion Matrix ---
cm_cnn = confusion_matrix(y_test, cnn_y_pred)
disp   = ConfusionMatrixDisplay(cm_cnn, display_labels=SPEAKERS)
disp.plot(cmap="Blues")
plt.title("VocalCanvas — CNN Confusion Matrix")
plt.savefig("confusion_matrix_cnn.png", dpi=150)
plt.show()

# --- Training curves ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history["accuracy"],     label="Train")
ax1.plot(history.history["val_accuracy"], label="Validation")
ax1.set_title("Accuracy"); ax1.legend()
ax2.plot(history.history["loss"],         label="Train")
ax2.plot(history.history["val_loss"],     label="Validation")
ax2.set_title("Loss"); ax2.legend()
plt.suptitle("VocalCanvas — CNN Training Curves")
plt.savefig("training_curves.png", dpi=150)
plt.show()

# --- Save ---
cnn_model.save(CNN_MODEL_PATH)
print(f"\nCNN saved → {CNN_MODEL_PATH}")


# =========================================================
# ██╗  ██╗      ███╗   ███╗███████╗ █████╗ ███╗   ██╗███████╗
# SECTION 2 — K-Means  (Unsupervised)
# =========================================================
print("\n" + "=" * 60)
print("SECTION 2 — K-Means Clustering  (Unsupervised)")
print("=" * 60)

# ---------------------------------------------------------
# Step 1 — Extract DSP features from raw audio
# ---------------------------------------------------------
print("\nExtracting DSP features from raw WAV files...")
print("(MFCCs + deltas + spectral features — 144 dims per chunk)")

X_feat_all, y_feat_all = load_kmeans_features(SPEAKERS)

if len(X_feat_all) == 0:
    print(
        "\n  ERROR: No features extracted. "
        "Ensure preprocess.py has been run and "
        f"'{TEMP_WAV_DIR}/' is not empty."
    )
    raise SystemExit(1)

print(
    f"\n  Total chunks (all speakers): {len(X_feat_all)}"
    f"  Feature shape: {X_feat_all.shape}"
)

# ---------------------------------------------------------
# Step 2 — Train / test split for evaluation
# ---------------------------------------------------------
X_km_train, X_km_test, y_km_train, y_km_test = train_test_split(
    X_feat_all, y_feat_all,
    test_size=0.20,
    random_state=42,
    stratify=y_feat_all
)

print(
    f"  K-Means split → "
    f"train: {len(X_km_train)} | test: {len(X_km_test)}"
)

# ---------------------------------------------------------
# Step 3 — Scale features
#
# K-Means uses Euclidean distance.  Without scaling, a
# feature with range [0, 8000] (centroid Hz) dominates
# a feature with range [-1, 1] (normalised MFCC) by
# a factor of ~8000×.  StandardScaler removes this bias.
#
# CRITICAL: fit ONLY on training data, transform both sets.
# ---------------------------------------------------------
print("\nScaling features with StandardScaler...")

scaler        = StandardScaler()
X_km_train_sc = scaler.fit_transform(X_km_train)
X_km_test_sc  = scaler.transform(X_km_test)

# ---------------------------------------------------------
# Step 4 — Elbow + Silhouette analysis
# ---------------------------------------------------------
print("\nRunning optimal-k analysis...")

analyze_k_selection(
    X_km_train_sc,
    SPEAKERS,
    output_path="kmeans_k_analysis.png"
)

# ---------------------------------------------------------
# Step 5 — Fit K-Means
# ---------------------------------------------------------
print(f"\nFitting K-Means (k={len(SPEAKERS)}, n_init=20)...")

kmeans = KMeans(
    n_clusters=len(SPEAKERS),
    n_init=20,
    max_iter=500,
    random_state=42
)
kmeans.fit(X_km_train_sc)

# ---------------------------------------------------------
# Step 6 — Build Cluster → Speaker mapping (majority vote)
# ---------------------------------------------------------
print("\nBuilding cluster → speaker mapping (majority vote)...")

train_cluster_ids = kmeans.labels_     # assigned during fit()
cluster_map       = {}                 # {cluster_id: speaker_name}

for cluster_id in range(len(SPEAKERS)):

    mask = (train_cluster_ids == cluster_id)

    if not mask.any():
        cluster_map[cluster_id] = SPEAKERS[0]
        print(
            f"  ⚠  Cluster {cluster_id} is empty — "
            f"mapped to '{SPEAKERS[0]}' by default."
        )
        continue

    labels_in_cluster = y_km_train[mask]
    vote_counts       = Counter(labels_in_cluster.tolist())
    dominant_label    = vote_counts.most_common(1)[0][0]
    dominant_speaker  = SPEAKERS[dominant_label]

    cluster_map[cluster_id] = dominant_speaker

    breakdown = "  |  ".join(
        f"{SPEAKERS[lbl]}: {cnt} ({cnt/mask.sum()*100:.0f}%)"
        for lbl, cnt in sorted(vote_counts.items())
    )

    print(
        f"  Cluster {cluster_id} → '{dominant_speaker}' "
        f"[{breakdown}]"
    )

# Warn on collision (two clusters pointing at the same speaker)
if len(set(cluster_map.values())) < len(SPEAKERS):
    print(
        "\n  ⚠  COLLISION: multiple clusters map to the same speaker. "
        "K-Means did not separate speakers cleanly. "
        "Consider collecting more training audio."
    )

# ---------------------------------------------------------
# Step 7 — Evaluate on test set
# ---------------------------------------------------------
print("\nEvaluating K-Means on test set...")

test_cluster_ids = kmeans.predict(X_km_test_sc)

km_y_pred = np.array([
    SPEAKERS.index(cluster_map[c])
    for c in test_cluster_ids
])

km_accuracy = np.mean(km_y_pred == y_km_test)

print(f"K-Means Test Accuracy: {km_accuracy * 100:.2f}%")
print("\nK-Means Classification Report:")
print(
    classification_report(
        y_km_test, km_y_pred,
        target_names=SPEAKERS,
        zero_division=0
    )
)

# --- Confusion Matrix ---
cm_km  = confusion_matrix(y_km_test, km_y_pred)
disp2  = ConfusionMatrixDisplay(cm_km, display_labels=SPEAKERS)
disp2.plot(cmap="Oranges")
plt.title("VocalCanvas — K-Means Confusion Matrix")
plt.savefig("confusion_matrix_kmeans.png", dpi=150)
plt.show()

# ---------------------------------------------------------
# Step 8 — Side-by-side comparison
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(f"  CNN    (supervised)    : {test_acc    * 100:.2f}%")
print(f"  K-Means (unsupervised) : {km_accuracy * 100:.2f}%")
print("=" * 60)

# ---------------------------------------------------------
# Step 9 — Save K-Means bundle
# Everything predict.py needs is stored in ONE pkl:
#   kmeans      — fitted KMeans
#   scaler      — fitted StandardScaler (MUST match training)
#   cluster_map — {int → speaker_name}
#   speakers    — ordered speaker list
#   n_features  — sanity-check at load time
# ---------------------------------------------------------
kmeans_bundle = {
    "kmeans":      kmeans,
    "scaler":      scaler,
    "cluster_map": cluster_map,
    "speakers":    SPEAKERS,
    "n_features":  PROC.n_features,   # 144
}

joblib.dump(kmeans_bundle, KMEANS_PKL_PATH)

print(f"\nK-Means bundle saved → {KMEANS_PKL_PATH}")
print(f"  Cluster map : {cluster_map}")
print(f"  n_features  : {PROC.n_features}")

print("\n✅  Training complete.")
print(
    f"   CNN model   → {CNN_MODEL_PATH}\n"
    f"   K-Means pkg → {KMEANS_PKL_PATH}\n"
    f"   K plot      → kmeans_k_analysis.png"
)