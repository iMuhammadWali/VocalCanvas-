import librosa
import numpy as np
import os
import subprocess
import shutil
from sklearn.model_selection import train_test_split

VOICES_DIR = "voices"
OUTPUT_DIR = "spectrograms"
TEMP_WAV_DIR = "temp_wavs"

CHUNK_DURATION = 3
SAMPLE_RATE = 22050
N_MELS = 128


# =========================================
# Convert any audio/video file to WAV
# =========================================
SUPPORTED_EXTENSIONS = (
    ".wav",
    ".mp3",
    ".mp4",
    ".ogg",
    ".flac",
    ".m4a",
    ".aac",
    ".wma",
    ".opus",
    ".webm",
    ".mov",
    ".mkv"
)


def convert_to_wav(input_path, output_path):
    """
    Uses ffmpeg to convert audio/video files to wav.
    """

    command = [
        "ffmpeg",
        "-y",                 # overwrite existing
        "-i", input_path,     # input file
        "-ac", "1",           # mono audio
        "-ar", str(SAMPLE_RATE),
        output_path
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )


# =========================================
# Audio augmentation
# =========================================
def augment_audio(chunk, sr):
    augmented = [chunk]

    augmented.append(
        librosa.effects.pitch_shift(
            chunk,
            sr=sr,
            n_steps=2
        )
    )

    augmented.append(
        librosa.effects.pitch_shift(
            chunk,
            sr=sr,
            n_steps=-2
        )
    )

    augmented.append(
        librosa.effects.time_stretch(
            chunk,
            rate=1.1
        )
    )

    augmented.append(
        librosa.effects.time_stretch(
            chunk,
            rate=0.9
        )
    )

    noise = np.random.normal(0, 0.005, chunk.shape)
    augmented.append(chunk + noise)

    return augmented


# =========================================
# Convert chunk to mel spectrogram
# =========================================
def chunk_to_spectrogram(chunk):
    mel = librosa.feature.melspectrogram(
        y=chunk,
        sr=SAMPLE_RATE,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    denom = mel_db.max() - mel_db.min()

    if denom == 0:
        mel_norm = np.zeros_like(mel_db)
    else:
        mel_norm = (
            (mel_db - mel_db.min()) / denom
        )

    return mel_norm


# =========================================
# Process one speaker
# =========================================
def process_speaker(speaker_name):
    print(f"\nProcessing {speaker_name}...")

    speaker_dir = os.path.join(
        VOICES_DIR,
        speaker_name
    )

    temp_speaker_dir = os.path.join(
        TEMP_WAV_DIR,
        speaker_name
    )

    os.makedirs(temp_speaker_dir, exist_ok=True)

    # =====================================
    # Convert all supported files to wav
    # =====================================
    converted_wavs = []

    files = sorted(os.listdir(speaker_dir))

    for file_name in files:

        ext = os.path.splitext(file_name)[1].lower()

        if ext not in SUPPORTED_EXTENSIONS:
            continue

        input_path = os.path.join(
            speaker_dir,
            file_name
        )

        output_wav = os.path.join(
            temp_speaker_dir,
            f"{os.path.splitext(file_name)[0]}.wav"
        )

        try:
            print(f"  Converting {file_name} -> wav")

            convert_to_wav(
                input_path,
                output_wav
            )

            converted_wavs.append(output_wav)

        except Exception as e:
            print(f"  Failed converting {file_name}")
            print(f"  Error: {e}")

    if not converted_wavs:
        print("  No valid audio files found")
        return

    all_chunks = []

    # =====================================
    # Load wav files
    # =====================================
    for wav_path in converted_wavs:

        wav_file = os.path.basename(wav_path)

        print(f"  Loading {wav_file}...")

        audio, sr = librosa.load(
            wav_path,
            sr=SAMPLE_RATE
        )

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

        # Remove silent chunks
        chunks = [
            c for c in chunks
            if np.max(np.abs(c)) >= 0.01
        ]

        all_chunks.extend(chunks)

    print(f"  Total valid chunks: {len(all_chunks)}")

    # =====================================
    # Split BEFORE augmentation
    # =====================================
    indices = list(range(len(all_chunks)))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=42
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=42
    )

    counts = {
        "train": 0,
        "val": 0,
        "test": 0
    }

    # =====================================
    # Generate spectrograms
    # =====================================
    for split_name, split_indices in [
        ("train", train_idx),
        ("val", val_idx),
        ("test", test_idx)
    ]:

        out_dir = os.path.join(
            OUTPUT_DIR,
            split_name,
            speaker_name
        )

        os.makedirs(out_dir, exist_ok=True)

        for idx in split_indices:

            chunk = all_chunks[idx]

            if split_name == "train":
                versions = augment_audio(chunk, sr)
            else:
                versions = [chunk]

            for aug_idx, aug_chunk in enumerate(versions):

                aug_chunk = aug_chunk[:chunk_samples]

                if len(aug_chunk) < chunk_samples:
                    continue

                spec = chunk_to_spectrogram(
                    aug_chunk
                )

                out_path = os.path.join(
                    out_dir,
                    f"chunk_{idx:05d}_aug{aug_idx}.npy"
                )

                np.save(out_path, spec)

                counts[split_name] += 1

    print(
        f"  Train: {counts['train']} | "
        f"Val: {counts['val']} | "
        f"Test: {counts['test']}"
    )


# =========================================
# Clear old folders
# =========================================
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
    print("Cleared old spectrograms")

if os.path.exists(TEMP_WAV_DIR):
    shutil.rmtree(TEMP_WAV_DIR)

os.makedirs(TEMP_WAV_DIR, exist_ok=True)


# =========================================
# Process speakers
# =========================================
for speaker in [
    "ahmed",
    "zoha",
]:
    process_speaker(speaker)

print("\nPreprocessing complete!")