from pathlib import Path
import librosa
import soundfile as sf
import nemo.collections.asr as nemo_asr
import torch
import pandas as pd


def load_asr_model():
    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v2"
    )

    # 🔥 CRITICAL FIX
    if hasattr(model, "change_decoding_strategy"):
        model.change_decoding_strategy(
            decoding_cfg={
                "use_cuda_graph": False
            }
        )

    return model


def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int(seconds // 60)
    secs = round(seconds % 60, 2)
    return f"{hours:02}:{minutes:02}:{secs:05.2f}"


def transcribe_isolated_voice(
    audio_path: Path,
    temp_chunk_dir: Path,
    asr_model,
    chunk_duration_sec: int = 30
) -> pd.DataFrame:

    audio_data, sample_rate = librosa.load(
        audio_path, sr=None, mono=True
    )

    chunk_samples = int(chunk_duration_sec * sample_rate)
    chunks = [
        audio_data[i:i + chunk_samples]
        for i in range(0, len(audio_data), chunk_samples)
    ]

    temp_chunk_dir.mkdir(parents=True, exist_ok=True)

    segments = []

    for i, chunk in enumerate(chunks):
        chunk_path = temp_chunk_dir / f"chunk_{i:04d}.wav"
        sf.write(chunk_path, chunk, sample_rate)

        output = asr_model.transcribe(
            [str(chunk_path)], timestamps=True
        )

        offset = i * chunk_duration_sec
        for seg in output[0].timestamp["segment"]:
            segments.append({
                "filename": audio_path.name,
                "start": format_time(seg["start"] + offset),
                "end": format_time(seg["end"] + offset),
                "transcription": seg["segment"]
            })

        torch.cuda.empty_cache()

    return pd.DataFrame(segments)

