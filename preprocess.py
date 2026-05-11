# # import librosa
# # import numpy as np
# # import os
# # import subprocess

# # # Config
# # VOICES_DIR = "voices"
# # OUTPUT_DIR = "spectrograms"
# # CHUNK_DURATION = 3  # seconds
# # SAMPLE_RATE = 22050
# # N_MELS = 128

# # def convert_to_wav(speaker_name):
# #     input_path = os.path.join(VOICES_DIR, speaker_name, "full_length.webm")
# #     output_path = os.path.join(VOICES_DIR, speaker_name, "full_length.wav")
    
# #     if os.path.exists(output_path):
# #         print(f"  WAV already exists for {speaker_name}, skipping conversion")
# #         return output_path
    
# #     print(f"  Converting {speaker_name} webm to wav...")
# #     subprocess.run([
# #         "ffmpeg", "-i", input_path,
# #         "-ar", str(SAMPLE_RATE),
# #         "-ac", "1",  # mono
# #         output_path
# #     ], check=True)
# #     print(f"  Conversion done!")
# #     return output_path

# # def process_speaker(speaker_name):
# #     print(f"\nProcessing {speaker_name}...")
    
# #     # Convert webm to wav first
# #     wav_path = convert_to_wav(speaker_name)
    
# #     output_dir = os.path.join(OUTPUT_DIR, speaker_name)
# #     os.makedirs(output_dir, exist_ok=True)

# #     print(f"  Loading audio...")
# #     audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE)

# #     # Split into 3s chunks
# #     chunk_samples = CHUNK_DURATION * SAMPLE_RATE
# #     chunks = [audio[i:i+chunk_samples] 
# #               for i in range(0, len(audio) - chunk_samples, chunk_samples)]

# #     print(f"  Total chunks: {len(chunks)}")

# #     saved = 0
# #     for idx, chunk in enumerate(chunks):
# #         # Skip silent chunks
# #         if np.max(np.abs(chunk)) < 0.01:
# #             continue

# #         # Convert to Mel spectrogram
# #         mel = librosa.feature.melspectrogram(y=chunk, sr=SAMPLE_RATE, n_mels=N_MELS)
# #         mel_db = librosa.power_to_db(mel, ref=np.max)

# #         # Normalize to [0, 1]
# #         mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())

# #         # Save as numpy array
# #         out_path = os.path.join(output_dir, f"chunk_{idx:04d}.npy")
# #         np.save(out_path, mel_norm)
# #         saved += 1

# #     print(f"  Saved {saved} spectrograms for {speaker_name}")

# # # Process both speakers
# # for speaker in ["nanami", "gojo"]:
# #     process_speaker(speaker)

# # print("\nPreprocessing complete!")
# # print(f"Check your {OUTPUT_DIR}/ folder")

# import librosa
# import numpy as np
# import os
# import subprocess

# VOICES_DIR = "voices"
# OUTPUT_DIR = "spectrograms"
# CHUNK_DURATION = 3
# SAMPLE_RATE = 22050
# N_MELS = 128

# def convert_to_wav(speaker_name):
#     input_path = os.path.join(VOICES_DIR, speaker_name, "full_length.webm")
#     output_path = os.path.join(VOICES_DIR, speaker_name, "full_length.wav")
#     if os.path.exists(output_path):
#         print(f"  WAV already exists for {speaker_name}, skipping")
#         return output_path
#     print(f"  Converting {speaker_name} to wav...")
#     subprocess.run([
#         "ffmpeg", "-i", input_path,
#         "-ar", str(SAMPLE_RATE),
#         "-ac", "1",
#         output_path
#     ], check=True)
#     return output_path

# def augment_audio(chunk, sr):
#     augmented = [chunk]  # original

#     # Pitch shift up
#     augmented.append(librosa.effects.pitch_shift(chunk, sr=sr, n_steps=2))
#     # Pitch shift down
#     augmented.append(librosa.effects.pitch_shift(chunk, sr=sr, n_steps=-2))
#     # Time stretch faster
#     augmented.append(librosa.effects.time_stretch(chunk, rate=1.1))
#     # Time stretch slower
#     augmented.append(librosa.effects.time_stretch(chunk, rate=0.9))
#     # Add slight noise
#     noise = np.random.normal(0, 0.005, chunk.shape)
#     augmented.append(chunk + noise)

#     return augmented

# def chunk_to_spectrogram(chunk):
#     mel = librosa.feature.melspectrogram(y=chunk, sr=SAMPLE_RATE, n_mels=N_MELS)
#     mel_db = librosa.power_to_db(mel, ref=np.max)
#     mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
#     return mel_norm

# def process_speaker(speaker_name):
#     print(f"\nProcessing {speaker_name}...")
#     wav_path = convert_to_wav(speaker_name)
#     output_dir = os.path.join(OUTPUT_DIR, speaker_name)
#     os.makedirs(output_dir, exist_ok=True)

#     print(f"  Loading audio...")
#     audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE)

#     chunk_samples = CHUNK_DURATION * SAMPLE_RATE
#     chunks = [audio[i:i+chunk_samples]
#               for i in range(0, len(audio) - chunk_samples, chunk_samples)]

#     print(f"  Original chunks: {len(chunks)}")

#     saved = 0
#     for idx, chunk in enumerate(chunks):
#         if np.max(np.abs(chunk)) < 0.01:
#             continue

#         # Get original + augmented versions
#         versions = augment_audio(chunk, sr)

#         for aug_idx, aug_chunk in enumerate(versions):
#             # Make sure all chunks are same length
#             aug_chunk = aug_chunk[:chunk_samples]
#             if len(aug_chunk) < chunk_samples:
#                 continue

#             spec = chunk_to_spectrogram(aug_chunk)
#             out_path = os.path.join(
#                 output_dir, f"chunk_{idx:04d}_aug{aug_idx}.npy")
#             np.save(out_path, spec)
#             saved += 1

#     print(f"  Saved {saved} spectrograms for {speaker_name} (with augmentation)")

# # Clear old spectrograms first
# import shutil
# if os.path.exists(OUTPUT_DIR):
#     shutil.rmtree(OUTPUT_DIR)
#     print("Cleared old spectrograms")

# for speaker in ["nanami", "gojo"]:
#     process_speaker(speaker)

# print("\nPreprocessing complete!")

import librosa
import numpy as np
import os
import subprocess
import shutil
from sklearn.model_selection import train_test_split

VOICES_DIR = "voices"
OUTPUT_DIR = "spectrograms"
CHUNK_DURATION = 3
SAMPLE_RATE = 22050
N_MELS = 128

# I need to understand what is augmentation this.
def augment_audio(chunk, sr):
    augmented = [chunk]
    augmented.append(librosa.effects.pitch_shift(chunk, sr=sr, n_steps=2))
    augmented.append(librosa.effects.pitch_shift(chunk, sr=sr, n_steps=-2))
    augmented.append(librosa.effects.time_stretch(chunk, rate=1.1))
    augmented.append(librosa.effects.time_stretch(chunk, rate=0.9))
    noise = np.random.normal(0, 0.005, chunk.shape)
    augmented.append(chunk + noise)
    return augmented

def chunk_to_spectrogram(chunk):
    mel = librosa.feature.melspectrogram(y=chunk, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
    denom = mel_db.max() - mel_db.min()

    if denom == 0:
        mel_norm = np.zeros_like(mel_db)
    else:
        mel_norm = (mel_db - mel_db.min()) / denom
    return mel_norm

def process_speaker(speaker_name):
    print(f"\nProcessing {speaker_name}...")

    speaker_dir = os.path.join("voices", speaker_name)

    # Get all wav files
    wav_files = sorted([
        f for f in os.listdir(speaker_dir)
        if f.endswith(".wav")
    ])

    if not wav_files:
        print("  No wav files found")
        return

    all_chunks = []

    # Load every wav file
    for wav_file in wav_files:
        wav_path = os.path.join(speaker_dir, wav_file)

        print(f"  Loading {wav_file}...")
        audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE)

        chunk_samples = CHUNK_DURATION * SAMPLE_RATE

        chunks = [
            audio[i:i+chunk_samples]
            for i in range(0, len(audio) - chunk_samples, chunk_samples)
        ]

        # Remove silent chunks
        chunks = [c for c in chunks if np.max(np.abs(c)) >= 0.01]

        all_chunks.extend(chunks)

    print(f"  Total valid chunks: {len(all_chunks)}")

    # Split BEFORE augmentation
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

    counts = {"train": 0, "val": 0, "test": 0}

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

                spec = chunk_to_spectrogram(aug_chunk)

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

# Clear old spectrograms
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
    print("Cleared old spectrograms")

for speaker in ["abi", "kenjiro_tsuda", "megumi_hayashibara"]:
    process_speaker(speaker)

print("\nPreprocessing complete!") 