import subprocess
import shutil
from pathlib import Path


def isolate_voice_with_demucs(
    input_path: Path,
    output_dir: Path,
    demucs_temp_out: Path,
    model: str = "mdx_extra_q"
):
    output_dir.mkdir(parents=True, exist_ok=True)
    demucs_temp_out.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["demucs", "--name", model, "--out", str(demucs_temp_out), str(input_path)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return None, result.stderr

    input_basename = input_path.stem
    vocals_path = (
        demucs_temp_out / model / input_basename / "vocals.wav"
    )

    if not vocals_path.exists():
        return None, f"vocals.wav not found for {input_basename}"

    final_output_path = output_dir / f"{input_basename}_final.wav"
    shutil.copy(vocals_path, final_output_path)

    return final_output_path, None


def process_directory_of_denoised_files(
    input_dir: Path,
    output_dir: Path,
    demucs_temp_out: Path,
    suffix="_denoised.wav"
):
    processed = []
    errors = []

    for audio_path in sorted(input_dir.iterdir()):
        if not audio_path.name.lower().endswith(suffix.lower()):
            continue

        out, err = isolate_voice_with_demucs(
            input_path=audio_path,
            output_dir=output_dir,
            demucs_temp_out=demucs_temp_out
        )

        if err:
            errors.append((audio_path.name, err))
        else:
            processed.append(out.name)

    return processed, errors

