import librosa
import numpy as np
import tensorflow as tf
import sys
import os
import subprocess
import cv2

# =========================================================
# Config
# =========================================================
MODEL_PATH = "models/vocalcanvas_sanity.keras"

SPEAKERS = [
    "ahmed",
    "zoha"
]

CHUNK_DURATION = 3
SAMPLE_RATE = 22050
N_MELS = 128

IMG_SIZE = 128

# =========================================================
# Load Model
# =========================================================
def load_model():

    print("Loading model...")

    return tf.keras.models.load_model(
        MODEL_PATH
    )

# =========================================================
# Convert any file to wav
# =========================================================
def convert_to_wav(input_path):

    base = os.path.splitext(
        input_path
    )[0]

    output_path = (
        base + "_converted.wav"
    )

    print("Converting to wav...")

    subprocess.run([

        "ffmpeg",

        "-i", input_path,

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
# Audio -> Spectrogram Image
# =========================================================
def audio_to_spectrogram(chunk):

    mel = librosa.feature.melspectrogram(

        y=chunk,
        sr=SAMPLE_RATE,
        n_mels=N_MELS

    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Normalize 0-1
    mel_norm = (

        (mel_db - mel_db.min()) /

        (mel_db.max() - mel_db.min() + 1e-9)

    )

    # Convert to image
    mel_img = (
        mel_norm * 255
    ).astype(np.uint8)

    # Resize to match training
    mel_img = cv2.resize(
        mel_img,
        (IMG_SIZE, IMG_SIZE)
    )

    # Normalize again
    mel_img = (
        mel_img.astype(np.float32) / 255.0
    )

    return mel_img

# =========================================================
# Predict
# =========================================================
def predict_file(audio_path, model):

    # =====================================================
    # Convert if not wav
    # =====================================================
    ext = os.path.splitext(
        audio_path
    )[1].lower()

    if ext != ".wav":

        audio_path = convert_to_wav(
            audio_path
        )

    print(f"\nLoading audio: {audio_path}")

    audio, sr = librosa.load(

        audio_path,
        sr=SAMPLE_RATE

    )

    # =====================================================
    # Boost quiet audio
    # =====================================================
    max_amp = np.max(np.abs(audio))

    if max_amp < 0.1:

        audio = audio / (
            max_amp + 1e-9
        )

        print(
            "  (Audio boosted — original was very quiet)"
        )

    # =====================================================
    # Split into chunks
    # =====================================================
    chunk_samples = (
        CHUNK_DURATION * SAMPLE_RATE
    )

    chunks = [

        audio[i:i + chunk_samples]

        for i in range(
            0,
            len(audio) - chunk_samples,
            chunk_samples
        )

    ]

    print(f"Total chunks: {len(chunks)}")

    votes = {

        speaker: 0
        for speaker in SPEAKERS

    }

    confidences = []

    # =====================================================
    # Predict each chunk
    # =====================================================
    for idx, chunk in enumerate(chunks):

        # Skip silence
        if np.max(np.abs(chunk)) < 0.02:

            print(
                f"  Chunk {idx:03d}: "
                f"SKIPPED (silence)"
            )

            continue

        # Convert to spectrogram
        spec = audio_to_spectrogram(
            chunk
        )

        # Add channel dimension
        spec = np.expand_dims(
            spec,
            axis=-1
        )

        # Add batch dimension
        spec = np.expand_dims(
            spec,
            axis=0
        )

        # Predict
        probs = model.predict(
            spec,
            verbose=0
        )[0]

        pred_label = np.argmax(
            probs
        )

        confidence = probs[
            pred_label
        ]

        predicted_speaker = SPEAKERS[
            pred_label
        ]

        votes[predicted_speaker] += 1

        confidences.append(
            confidence
        )

        print(

            f"  Chunk {idx:03d}: "

            f"{predicted_speaker} "

            f"({confidence * 100:.1f}%)"

        )

    # =====================================================
    # Final results
    # =====================================================
    print("\n--- Results ---")

    total = sum(votes.values())

    if total == 0:

        print(
            "No valid chunks detected."
        )

        return

    for speaker, count in votes.items():

        print(

            f"  {speaker}: "

            f"{count}/{total} chunks "

            f"({count / total * 100:.1f}%)"

        )

    winner = max(
        votes,
        key=votes.get
    )

    avg_confidence = (
        np.mean(confidences) * 100
    )

    print(
        f"\n  Predicted speaker: "
        f"{winner.upper()}"
    )

    print(
        f"  Average confidence: "
        f"{avg_confidence:.1f}%"
    )

# =========================================================
# Main
# =========================================================
if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage: python predict.py <audio_file>"
        )

        print(
            "Example:"
        )

        print(
            "python predict.py test.mp3"
        )

        sys.exit(1)

    audio_path = sys.argv[1]

    if not os.path.exists(audio_path):

        print(
            f"File not found: {audio_path}"
        )

        sys.exit(1)

    model = load_model()

    predict_file(
        audio_path,
        model
    )