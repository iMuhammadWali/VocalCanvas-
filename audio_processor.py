"""
audio_processor.py — VocalCanvas Unified Audio Processor
=========================================================
Single source of truth for all DSP operations.
Import this in BOTH preprocess.py and predict.py.

Public API
----------
chunk_to_spectrogram(chunk)
    Raw audio chunk  →  (IMG_SIZE, IMG_SIZE, 1) float32 [0,1]   ← CNN input

chunk_to_kmeans_features(chunk)
    Raw audio chunk  →  (144,) float32                          ← K-Means input
    Computes MFCCs + deltas + spectral features from raw audio.
    Never goes through the image pipeline — full acoustic fidelity.

    Feature vector (144 dims):
        MFCCs × 20          → mean + std  =  40
        Delta-MFCCs × 20    → mean + std  =  40
        Delta²-MFCCs × 20   → mean + std  =  40
        Spectral Centroid   → mean + std  =   2
        Spectral Bandwidth  → mean + std  =   2
        Spectral Contrast×7 → mean + std  =  14
        Spectral Roll-off   → mean + std  =   2
        Zero Crossing Rate  → mean + std  =   2
        RMS Energy          → mean + std  =   2
                                             ───
                             Total         = 144

is_silent(chunk)         → bool
split_audio(audio)       → list[np.ndarray]
"""

import librosa
import numpy as np
from PIL import Image


class UnifiedAudioProcessor:
    """
    Encapsulates every DSP step that the model depends on.
    All parameters that affect spectrogram shape or value range live here.

    Training contract
    -----------------
    - 3-second chunks at 22 050 Hz  →  66 150 samples
    - Mel spectrogram: 128 bands, librosa defaults for n_fft / hop_length / window
    - power_to_db with ref=np.max
    - Local min-max normalization to [0, 1] in float32  ← pure numpy, no matplotlib
    - PIL resize to (IMG_SIZE, IMG_SIZE) with BILINEAR  ← same resampler both ways
    - Output shape: (IMG_SIZE, IMG_SIZE, 1)
    - Silence threshold for skipping chunks: MAX_ABS_THRESHOLD

    Never amplitude-normalize the audio before calling chunk_to_spectrogram().
    """

    # ------------------------------------------------------------------
    # DSP parameters — change these in ONE place only
    # ------------------------------------------------------------------
    SAMPLE_RATE: int      = 22_050
    CHUNK_DURATION: int   = 3           # seconds
    N_MELS: int           = 128
    N_MFCC: int           = 20          # MFCCs extracted for K-Means
    # n_fft, hop_length, window: librosa defaults (2048, 512, hann)

    IMG_SIZE: int         = 128         # square output image side length

    # Chunks whose peak amplitude is below this are considered silence
    SILENCE_THRESHOLD: float = 0.01

    # ------------------------------------------------------------------
    # Derived constants
    # ------------------------------------------------------------------
    @property
    def chunk_samples(self) -> int:
        """Exact number of samples in one chunk (integer, no rounding needed)."""
        return self.CHUNK_DURATION * self.SAMPLE_RATE   # 66 150

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_silent(self, chunk: np.ndarray) -> bool:
        """Return True if the chunk should be skipped as silence."""
        return np.max(np.abs(chunk)) < self.SILENCE_THRESHOLD

    def chunk_to_spectrogram(self, chunk: np.ndarray) -> np.ndarray:
        """
        Convert a 1-D audio chunk (float32, chunk_samples long) to a
        normalized spectrogram ready for model input.

        Returns
        -------
        np.ndarray  shape (IMG_SIZE, IMG_SIZE, 1), dtype float32, range [0, 1]
        """
        # 1. Mel spectrogram (uses librosa defaults: n_fft=2048, hop_length=512, hann)
        mel = librosa.feature.melspectrogram(
            y=chunk,
            sr=self.SAMPLE_RATE,
            n_mels=self.N_MELS,
        )

        # 2. Power → dB  (ref=np.max → range is approximately [-80, 0] dB)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # 3. Local min-max normalization  →  [0.0, 1.0]
        #    Using a small epsilon to guard against all-zero (silence) chunks.
        denom = mel_db.max() - mel_db.min()
        mel_norm = (mel_db - mel_db.min()) / (denom + 1e-9)   # float64

        # 4. Resize to (IMG_SIZE, IMG_SIZE) using PIL BILINEAR
        #    PIL expects uint8 for most modes; work in float via mode='F'.
        pil_img = Image.fromarray(mel_norm.astype(np.float32), mode='F')
        pil_img = pil_img.resize(
            (self.IMG_SIZE, self.IMG_SIZE),
            resample=Image.BILINEAR,
        )

        # 5. Back to numpy float32, add channel dimension
        img_array = np.array(pil_img, dtype=np.float32)        # (128, 128)
        img_array = np.expand_dims(img_array, axis=-1)         # (128, 128, 1)

        return img_array

    # 144 = (20+20+20)*2 + (1+1+7+1+1+1)*2
    @property
    def n_features(self) -> int:
        return 144

    def chunk_to_kmeans_features(self, chunk: np.ndarray) -> np.ndarray:
        """
        Extract a rich DSP feature vector from a raw audio chunk.
        Designed for K-Means clustering — never uses the image pipeline.

        Working directly from the time-domain signal preserves maximum
        acoustic fidelity.  The 144-dim vector is a superset of what
        speech processing research considers speaker-discriminative:

            MFCCs capture the vocal tract shape (timbre).
            Deltas capture rate-of-change (prosody/rhythm).
            Delta-deltas capture acceleration (speech dynamics).
            Centroid / Bandwidth capture spectral brightness.
            Contrast captures peak-vs-valley ratio (presence/noise).
            ZCR / RMS capture voiced/unvoiced energy.

        Parameters
        ----------
        chunk : np.ndarray  shape (chunk_samples,), dtype float32

        Returns
        -------
        np.ndarray  shape (144,), dtype float32
        """
        sr = self.SAMPLE_RATE

        def _stat(feat_2d: np.ndarray) -> np.ndarray:
            """(n, T) → [mean_over_T, std_over_T] → (2n,)"""
            return np.concatenate([
                feat_2d.mean(axis=1),
                feat_2d.std(axis=1)
            ])

        # --- MFCCs and temporal derivatives (120 dims) ---
        mfccs        = librosa.feature.mfcc(
            y=chunk, sr=sr, n_mfcc=self.N_MFCC
        )                                          # (20, T)
        delta_mfccs  = librosa.feature.delta(mfccs)           # (20, T)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)  # (20, T)

        # --- Spectral shape features (20 dims) ---
        centroid  = librosa.feature.spectral_centroid(
            y=chunk, sr=sr
        )                                          # (1, T)
        bandwidth = librosa.feature.spectral_bandwidth(
            y=chunk, sr=sr
        )                                          # (1, T)
        contrast  = librosa.feature.spectral_contrast(
            y=chunk, sr=sr
        )                                          # (7, T)
        rolloff   = librosa.feature.spectral_rolloff(
            y=chunk, sr=sr
        )                                          # (1, T)

        # --- Temporal features (4 dims) ---
        zcr = librosa.feature.zero_crossing_rate(chunk)  # (1, T)
        rms = librosa.feature.rms(y=chunk)               # (1, T)

        # --- Concatenate ---
        features = np.concatenate([
            _stat(mfccs),         #  40
            _stat(delta_mfccs),   #  40
            _stat(delta2_mfccs),  #  40
            _stat(centroid),      #   2
            _stat(bandwidth),     #   2
            _stat(contrast),      #  14
            _stat(rolloff),       #   2
            _stat(zcr),           #   2
            _stat(rms),           #   2
        ]).astype(np.float32)     # = 144

        return features

    def split_audio(self, audio: np.ndarray) -> list[np.ndarray]:
        """
        Split a 1-D audio array into non-overlapping CHUNK_DURATION chunks.
        The final partial chunk is discarded (same as training).

        Parameters
        ----------
        audio : np.ndarray   shape (N,), dtype float32

        Returns
        -------
        list of np.ndarray, each of length chunk_samples
        """
        n = self.chunk_samples
        return [
            audio[i : i + n]
            for i in range(0, len(audio) - n, n)
        ]
