"""
predict.py — VocalCanvas Dual-Model Inference
==============================================
Usage:
    # CNN (default)
    python predict.py <audio_file>
    python predict.py <audio_file> --model_type cnn

    # K-Means
    python predict.py <audio_file> --model_type kmeans

Both paths route audio through UnifiedAudioProcessor to guarantee
that the features the model sees at inference are identical to the
features it was trained on.
"""

import argparse
import librosa
import numpy as np
import os
import subprocess
import sys

import joblib
import tensorflow as tf

from audio_processor import UnifiedAudioProcessor

PROC = UnifiedAudioProcessor()

# =========================================================
# Config — must match train.py exactly
# =========================================================
CNN_MODEL_PATH  = "models/vocalcanvas_cnn.keras"   # written by train.py Section 1
KMEANS_PKL_PATH = "models/vocalcanvas_kmeans.pkl"

SPEAKERS = [
    "abi",
    "ahmed",
    "zoha"
]

# DSP constants from the single source of truth
SAMPLE_RATE    = PROC.SAMPLE_RATE
CHUNK_DURATION = PROC.CHUNK_DURATION
IMG_SIZE       = PROC.IMG_SIZE

# =========================================================
# Shared utility: convert any audio file to wav
# =========================================================
def convert_to_wav(input_path: str) -> str:

    base = os.path.splitext(input_path)[0]
    output_path = base + "_converted.wav"

    print("Converting to wav...")

    subprocess.run(
        [
            "ffmpeg",
            "-i",  input_path,
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            output_path,
            "-y"
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return output_path

# =========================================================
# Shared utility: load + chunk audio
# =========================================================
def load_and_chunk(audio_path: str):
    """
    Returns
    -------
    chunks : list[np.ndarray]  each of shape (chunk_samples,)
    """
    ext = os.path.splitext(audio_path)[1].lower()

    if ext != ".wav":
        audio_path = convert_to_wav(audio_path)

    print(f"\nLoading audio: {audio_path}")

    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)

    # No amplitude boost — training data was not boosted.
    # Quiet audio is handled naturally by per-chunk min-max
    # normalization inside UnifiedAudioProcessor.

    chunks = PROC.split_audio(audio)

    print(f"Total chunks: {len(chunks)}")

    return chunks

# =========================================================
# Shared utility: majority-vote result printer
# =========================================================
def print_results(votes: dict, confidences: list):

    print("\n--- Results ---")

    total = sum(votes.values())

    if total == 0:
        print("No valid (non-silent) chunks detected.")
        return

    for speaker, count in votes.items():
        print(
            f"  {speaker}: "
            f"{count}/{total} chunks "
            f"({count / total * 100:.1f}%)"
        )

    winner = max(votes, key=votes.get)

    print(f"\n  ▶  Predicted speaker : {winner.upper()}")

    if confidences:
        avg_conf = np.mean(confidences) * 100
        print(f"     Average confidence: {avg_conf:.1f}%")

# =========================================================
# ███████╗███████╗ ██████╗████████╗██╗ ██████╗ ███╗   ██╗
# ██╔════╝██╔════╝██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║
# ███████╗█████╗  ██║        ██║   ██║██║   ██║██╔██╗ ██║
# ╚════██║██╔══╝  ██║        ██║   ██║██║   ██║██║╚██╗██║
# ███████║███████╗╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║
# ╚══════╝╚══════╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
# CNN prediction path
# =========================================================
def load_cnn_model():
    print(f"Loading CNN model from {CNN_MODEL_PATH} ...")
    return tf.keras.models.load_model(CNN_MODEL_PATH)


def predict_cnn(audio_path: str, model) -> None:
    """
    Run chunk-by-chunk CNN inference.
    Each chunk → spectrogram → model.predict → softmax probabilities.
    Final answer: majority vote across all non-silent chunks.
    """
    chunks = load_and_chunk(audio_path)

    votes       = {s: 0 for s in SPEAKERS}
    confidences = []

    for idx, chunk in enumerate(chunks):

        if PROC.is_silent(chunk):
            print(f"  Chunk {idx:03d}: SKIPPED (silence)")
            continue

        # (128, 128, 1) → add batch dim → (1, 128, 128, 1)
        spec = PROC.chunk_to_spectrogram(chunk)
        spec = np.expand_dims(spec, axis=0)

        probs      = model.predict(spec, verbose=0)[0]
        pred_idx   = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        speaker    = SPEAKERS[pred_idx]

        votes[speaker] += 1
        confidences.append(confidence)

        print(
            f"  Chunk {idx:03d}: "
            f"{speaker} "
            f"({confidence * 100:.1f}%)"
        )

    print_results(votes, confidences)


# =========================================================
# ██╗  ██╗      ███╗   ███╗███████╗ █████╗ ███╗   ██╗███████╗
# ██║ ██╔╝      ████╗ ████║██╔════╝██╔══██╗████╗  ██║██╔════╝
# █████╔╝ █████╗██╔████╔██║█████╗  ███████║██╔██╗ ██║███████╗
# ██╔═██╗ ╚════╝██║╚██╔╝██║██╔══╝  ██╔══██║██║╚██╗██║╚════██║
# ██║  ██╗      ██║ ╚═╝ ██║███████╗██║  ██║██║ ╚████║███████║
# ╚═╝  ╚═╝      ╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝
# K-Means prediction path
# =========================================================
def load_kmeans_bundle():
    """
    Load the K-Means bundle saved by train.py.

    Returns
    -------
    dict with keys:
        kmeans      : fitted KMeans
        scaler      : fitted StandardScaler
        cluster_map : {int → speaker_name}
        speakers    : list[str]
        n_features  : int
    """
    print(f"Loading K-Means bundle from {KMEANS_PKL_PATH} ...")

    bundle = joblib.load(KMEANS_PKL_PATH)

    # Sanity check: feature dimensionality must match processor
    expected = PROC.n_features

    if bundle["n_features"] != expected:
        raise ValueError(
            f"Feature dimension mismatch: "
            f"bundle has {bundle['n_features']} "
            f"but processor expects {expected}. "
            f"Re-run train.py to rebuild the bundle."
        )

    print(
        "  Cluster map: "
        + str({k: v for k, v in bundle["cluster_map"].items()})
    )

    return bundle


def cluster_confidence(kmeans, scaler, feature_vec: np.ndarray) -> float:
    """
    Compute a [0, 1] confidence proxy for K-Means using the inverse
    of the normalised distance to the nearest centroid.

    K-Means has no native probability output.  We use:
        confidence = 1 / (1 + d_nearest / d_mean)
    where d_nearest is the distance to the winning centroid and
    d_mean is the mean distance to all centroids.

    A sample perfectly centred in its cluster → confidence ≈ 1.
    A sample equidistant from all centroids → confidence ≈ 0.5.
    """
    feat_scaled = scaler.transform(
        feature_vec.reshape(1, -1)
    )

    # Euclidean distance to every centroid
    diffs     = kmeans.cluster_centers_ - feat_scaled         # (k, n_feat)
    distances = np.linalg.norm(diffs, axis=1)                 # (k,)

    d_nearest = distances.min()
    d_mean    = distances.mean()

    confidence = 1.0 / (1.0 + d_nearest / (d_mean + 1e-9))

    return float(confidence)


def predict_kmeans(audio_path: str, bundle: dict) -> None:
    """
    Run chunk-by-chunk K-Means inference.
    Each chunk → 256-dim feature → scaler → kmeans.predict
              → cluster_map → speaker name.
    Final answer: majority vote across all non-silent chunks.
    """
    kmeans      = bundle["kmeans"]
    scaler      = bundle["scaler"]
    cluster_map = bundle["cluster_map"]

    chunks = load_and_chunk(audio_path)

    votes       = {s: 0 for s in SPEAKERS}
    confidences = []

    for idx, chunk in enumerate(chunks):

        if PROC.is_silent(chunk):
            print(f"  Chunk {idx:03d}: SKIPPED (silence)")
            continue

        # 144-dim DSP feature vector — identical pipeline to train.py
        # MFCCs + deltas + spectral features, all from raw audio.
        feat_vec = PROC.chunk_to_kmeans_features(chunk)

        # Scale using the scaler fitted on training data
        feat_scaled = scaler.transform(
            feat_vec.reshape(1, -1)
        )

        cluster_id = int(
            kmeans.predict(feat_scaled)[0]
        )

        speaker = cluster_map[cluster_id]

        # Distance-based confidence proxy
        confidence = cluster_confidence(kmeans, scaler, feat_vec)

        votes[speaker] += 1
        confidences.append(confidence)

        print(
            f"  Chunk {idx:03d}: "
            f"{speaker} "
            f"(cluster {cluster_id}, "
            f"conf {confidence * 100:.1f}%)"
        )

    print_results(votes, confidences)


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="VocalCanvas — Speaker Identification"
    )

    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to the audio file to identify."
    )

    parser.add_argument(
        "--model_type",
        type=str,
        choices=["cnn", "kmeans"],
        default="cnn",
        help=(
            "Which model to use for inference. "
            "'cnn' uses the deep learning model (default). "
            "'kmeans' uses the unsupervised clustering model."
        )
    )

    args = parser.parse_args()

    # ----------------------------------------------------------
    # Validate input file
    # ----------------------------------------------------------
    if not os.path.exists(args.audio_file):
        print(f"Error: File not found — {args.audio_file}")
        sys.exit(1)

    # ----------------------------------------------------------
    # Dispatch to the selected model
    # ----------------------------------------------------------
    print(f"\nModel type : {args.model_type.upper()}")
    print(f"Audio file : {args.audio_file}")
    print("-" * 50)

    if args.model_type == "cnn":

        model = load_cnn_model()
        predict_cnn(args.audio_file, model)

    elif args.model_type == "kmeans":

        bundle = load_kmeans_bundle()
        predict_kmeans(args.audio_file, bundle)