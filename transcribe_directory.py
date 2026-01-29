import shutil
from pathlib import Path
import pandas as pd

from transcription_stage import transcribe_isolated_voice

def transcribe_final_outputs(
    input_dir: Path,
    output_csv_dir: Path,
    temp_root: Path,
    asr_model
):
    output_csv_dir.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)

    all_dfs = []

    for audio_path in sorted(input_dir.iterdir()):
        if not audio_path.name.endswith("_final.wav"):
            continue

        clip_name = audio_path.stem
        temp_chunk_dir = temp_root / clip_name

        df = transcribe_isolated_voice(
            audio_path=audio_path,
            temp_chunk_dir=temp_chunk_dir,
            asr_model=asr_model
        )

        if not df.empty:
            per_file_csv = output_csv_dir / f"{clip_name}.csv"
            df.to_csv(per_file_csv, index=False)
            all_dfs.append(df)

        shutil.rmtree(temp_chunk_dir, ignore_errors=True)

    if all_dfs:
        consolidated_df = pd.concat(all_dfs, ignore_index=True)
        consolidated_csv = output_csv_dir / "ALL_TRANSCRIPTS_CONSOLIDATED.csv"
        consolidated_df.to_csv(consolidated_csv, index=False)
        return consolidated_csv, len(all_dfs)

    return None, 0

