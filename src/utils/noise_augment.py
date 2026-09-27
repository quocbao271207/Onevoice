"""
MediVoice Edge — Noise Augmentation Module
Mixes hospital noise into clean audio for robust ASR training.

Purpose:
- Train ASR models that work in real hospital environments (75+ dB noise)
- Augment VietMed and ViMedCSS datasets with hospital-specific noise
- Generate training data at various SNR levels (5-15 dB)

Noise sources:
- ESC-50 dataset (environment sounds)
- AudioSet (ambulance sirens, medical equipment)
- Custom hospital noise recordings
"""

import numpy as np
import logging
import random
from typing import List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


# Hospital-specific noise categories from ESC-50 / AudioSet
HOSPITAL_NOISE_CATEGORIES = [
    "siren",               # Ambulance sirens
    "breathing",           # Ventilator sounds
    "beep",                # Patient monitor beeps
    "footsteps",           # Staff walking
    "door_knocking",       # Door activity
    "keyboard_typing",     # Workstation sounds
    "crowd_noise",         # Waiting room chatter
    "air_conditioning",    # HVAC systems
    "phone_ringing",       # Hospital phones
    "clock_alarm",         # Timer / alarm sounds
]


def mix_audio_with_noise(
    clean_audio: np.ndarray,
    noise_audio: np.ndarray,
    snr_db: float = 10.0,
) -> np.ndarray:
    """
    Mix clean audio with noise at a specified SNR level.

    Args:
        clean_audio: Clean speech signal (float32)
        noise_audio: Noise signal (float32)
        snr_db: Signal-to-Noise Ratio in dB (5-15 dB for hospital conditions)

    Returns:
        Mixed audio signal (float32)
    """
    # Ensure noise is at least as long as clean audio
    if len(noise_audio) < len(clean_audio):
        # Loop the noise to match clean audio length
        repeats = (len(clean_audio) // len(noise_audio)) + 1
        noise_audio = np.tile(noise_audio, repeats)

    # Trim noise to match clean audio length
    noise_segment = noise_audio[:len(clean_audio)]

    # Calculate power of both signals
    clean_power = np.mean(clean_audio ** 2)
    noise_power = np.mean(noise_segment ** 2)

    # Avoid division by zero
    if noise_power < 1e-10:
        return clean_audio
    if clean_power < 1e-10:
        return noise_segment

    # Calculate required noise scaling factor for target SNR
    # SNR = 10 * log10(P_signal / P_noise)
    # P_noise_target = P_signal / 10^(SNR/10)
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    noise_scale = np.sqrt(target_noise_power / noise_power)

    # Mix signals
    mixed = clean_audio + noise_scale * noise_segment

    # Normalize to prevent clipping
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / max_val * 0.95

    return mixed.astype(np.float32)


def augment_dataset(
    audio_paths: List[str],
    noise_dir: str,
    output_dir: str,
    snr_range: Tuple[float, float] = (5.0, 15.0),
    augmentation_factor: int = 3,
    sample_rate: int = 16000,
):
    """
    Augment an entire dataset with hospital noise.

    For each clean audio file, creates `augmentation_factor` noisy versions
    at random SNR levels within the specified range.

    Args:
        audio_paths: List of paths to clean audio files
        noise_dir: Directory containing noise audio files
        output_dir: Directory to save augmented audio
        snr_range: Min and max SNR in dB
        augmentation_factor: Number of augmented versions per original
        sample_rate: Audio sample rate
    """
    import soundfile as sf

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    noise_dir_path = Path(noise_dir)
    noise_files = list(noise_dir_path.glob("*.wav")) + list(noise_dir_path.glob("*.flac"))

    if not noise_files:
        logger.error(f"No noise files found in {noise_dir}")
        return

    logger.info(
        f"Augmenting {len(audio_paths)} audio files with {len(noise_files)} noise sources "
        f"(factor={augmentation_factor}, SNR={snr_range[0]}-{snr_range[1]} dB)"
    )

    total_generated = 0
    for audio_path in audio_paths:
        try:
            clean_audio, sr = sf.read(audio_path)
            if sr != sample_rate:
                import scipy.signal
                num_samples = int(len(clean_audio) * sample_rate / sr)
                clean_audio = scipy.signal.resample(clean_audio, num_samples)

            clean_audio = clean_audio.astype(np.float32)

            # Generate augmented versions
            for aug_idx in range(augmentation_factor):
                # Random noise file and SNR
                noise_file = random.choice(noise_files)
                noise_audio, _ = sf.read(noise_file)
                noise_audio = noise_audio.astype(np.float32)

                snr = random.uniform(snr_range[0], snr_range[1])

                # Mix
                augmented = mix_audio_with_noise(clean_audio, noise_audio, snr_db=snr)

                # Save
                stem = Path(audio_path).stem
                out_file = output_path / f"{stem}_aug{aug_idx}_snr{snr:.0f}.wav"
                sf.write(str(out_file), augmented, sample_rate)
                total_generated += 1

        except Exception as e:
            logger.warning(f"Failed to augment {audio_path}: {e}")

    logger.info(f"Noise augmentation complete: {total_generated} files generated in {output_dir}")


def generate_synthetic_hospital_noise(
    duration_s: float = 60.0,
    sample_rate: int = 16000,
) -> np.ndarray:
    """
    Generate synthetic hospital background noise for testing.
    Combines multiple noise sources to simulate a busy ER environment.

    This is useful when real hospital noise recordings are not available.

    Args:
        duration_s: Duration of noise signal in seconds
        sample_rate: Sample rate

    Returns:
        Synthetic noise signal (float32)
    """
    num_samples = int(duration_s * sample_rate)
    noise = np.zeros(num_samples, dtype=np.float32)

    # Layer 1: Low-frequency HVAC rumble (50-200 Hz)
    t = np.linspace(0, duration_s, num_samples)
    hvac = 0.05 * np.sin(2 * np.pi * 60 * t + np.random.uniform(0, 2 * np.pi))
    noise += hvac

    # Layer 2: Random beeps (800-1200 Hz, intermittent)
    for _ in range(int(duration_s * 0.5)):  # ~0.5 beeps per second
        start = random.randint(0, max(1, num_samples - sample_rate))
        beep_duration = random.randint(int(0.1 * sample_rate), int(0.3 * sample_rate))
        freq = random.uniform(800, 1200)
        beep_t = np.linspace(0, beep_duration / sample_rate, beep_duration)
        beep = 0.03 * np.sin(2 * np.pi * freq * beep_t)
        end = min(start + beep_duration, num_samples)
        noise[start:end] += beep[:end - start]

    # Layer 3: Broadband background chatter
    chatter = np.random.normal(0, 0.01, num_samples).astype(np.float32)
    noise += chatter

    # Layer 4: Occasional footstep-like impacts
    for _ in range(int(duration_s * 2)):  # ~2 footsteps per second
        start = random.randint(0, max(1, num_samples - 1000))
        impact = np.exp(-np.linspace(0, 10, 1000)) * 0.04 * random.uniform(0.5, 1.5)
        end = min(start + 1000, num_samples)
        noise[start:end] += impact[:end - start]

    return noise.astype(np.float32)
