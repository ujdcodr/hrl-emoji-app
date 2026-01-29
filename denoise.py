import librosa
import noisereduce as nr
import soundfile as sf
from pathlib import Path

def denoise_directory(
    input_dir: Path,
    output_dir: Path,
    extensions=(".wav", ".WAV")
):
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_files = []

    for audio_path in sorted(input_dir.iterdir()):
        if audio_path.suffix not in extensions:
            continue

        # Load audio
        y, sr = librosa.load(audio_path, sr=None)

        # Denoise
        y_denoised = nr.reduce_noise(y=y, sr=sr)

        # Output filename
        out_path = output_dir / f"{audio_path.stem}_denoised{audio_path.suffix}"

        # Save
        sf.write(out_path, y_denoised, sr)

        processed_files.append(out_path.name)

    return processed_files

