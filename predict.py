import librosa
import numpy as np
import tensorflow as tf
import sys
import os
import subprocess

# Config
MODEL_PATH = "models/vocalcanvas_sanity.keras"
SPEAKERS = ["Kenjiro Tsuda", "Yuichi Nakamura"]
CHUNK_DURATION = 3
SAMPLE_RATE = 22050
N_MELS = 128

def load_model():
    print("Loading model...")
    return tf.keras.models.load_model(MODEL_PATH)

def convert_to_wav(input_path):
    base = os.path.splitext(input_path)[0]
    output_path = base + "_converted.wav"
    print(f"Converting to wav...")
    subprocess.run([
        "ffmpeg", "-i", input_path,
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        output_path,
        "-y"
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

def audio_to_spectrogram(chunk):
    mel = librosa.feature.melspectrogram(y=chunk, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
    return mel_norm

def predict_file(audio_path, model):
    # Convert to wav if needed
    ext = os.path.splitext(audio_path)[1].lower()
    if ext != ".wav":
        audio_path = convert_to_wav(audio_path)

    print(f"\nLoading audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    audio = audio / (np.max(np.abs(audio)) + 1e-9)

    chunk_samples = CHUNK_DURATION * SAMPLE_RATE
    chunks = [audio[i:i+chunk_samples]
              for i in range(0, len(audio) - chunk_samples, chunk_samples)]

    print(f"Total chunks: {len(chunks)}")

    votes = {speaker: 0 for speaker in SPEAKERS}
    confidences = []

    for idx, chunk in enumerate(chunks):
        if np.max(np.abs(chunk)) < 0.01:
            continue

        spec = audio_to_spectrogram(chunk)
        spec = spec[np.newaxis, ..., np.newaxis]

        probs = model.predict(spec, verbose=0)[0]
        pred_label = np.argmax(probs)
        confidence = probs[pred_label]

        votes[SPEAKERS[pred_label]] += 1
        confidences.append(confidence)

        print(f"  Chunk {idx:03d}: {SPEAKERS[pred_label]} ({confidence*100:.1f}%)")

    print("\n--- Results ---")
    total = sum(votes.values())
    for speaker, count in votes.items():
        print(f"  {speaker}: {count}/{total} chunks ({count/total*100:.1f}%)")

    winner = max(votes, key=votes.get)
    avg_confidence = np.mean(confidences) * 100
    print(f"\n  Predicted speaker: {winner.upper()}")
    print(f"  Average confidence: {avg_confidence:.1f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_audio_file>")
        print("Example: python predict.py voices/nanami/full_length.wav")
        sys.exit(1)

    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        sys.exit(1)

    model = load_model()
    predict_file(audio_path, model)