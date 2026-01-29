import streamlit as st
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
from denoise import denoise_directory
from demucs_stage import process_directory_of_denoised_files
from transcription_stage import load_asr_model
from transcribe_directory import transcribe_final_outputs
from irrelevance_stage import load_models, annotate_consolidated_csv
from sentiment_stage import (
    load_sentiment_pipeline,
    run_sentiment_on_consolidated_csv,
)

INPUT_ROOT = Path("inputs")
OUTPUT_ROOT = Path("outputs")
DEMUCS_TEMP = Path("demucs_tmp")

TRANSCRIPT_ROOT = Path("transcripts")
ASR_TEMP = Path("asr_chunks")

TRANSCRIPT_ROOT.mkdir(exist_ok=True)
ASR_TEMP.mkdir(exist_ok=True)

MODELS_ROOT = Path("./")

EMOJI_DIR = Path("Emojis")

for p in [INPUT_ROOT, OUTPUT_ROOT, DEMUCS_TEMP]:
    p.mkdir(exist_ok=True)

st.set_page_config(page_title="Audio Denoising Pipeline")
st.title("🎧 Two-Stage Audio Denoising Pipeline")

uploaded_files = st.file_uploader(
    "Upload WAV files",
    type=["wav", "WAV"],
    accept_multiple_files=True
)

if uploaded_files and st.button("Submit Processing Job"):
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    st.session_state["job_complete"] = False
    input_dir = INPUT_ROOT / f"job_{job_id}"
    stage1_dir = OUTPUT_ROOT / f"job_{job_id}_stage1"
    stage2_dir = OUTPUT_ROOT / f"job_{job_id}_stage2"

    input_dir.mkdir(parents=True)
    stage1_dir.mkdir(parents=True)
    stage2_dir.mkdir(parents=True)

    # Save uploads
    for f in uploaded_files:
        with open(input_dir / f.name, "wb") as out:
            out.write(f.read())

    # --------------------
    # Stage 1: NR denoise
    # --------------------
    with st.spinner("Stage 1: Noise reduction..."):
        denoised_files = denoise_directory(
            input_dir=input_dir,
            output_dir=stage1_dir
        )

    st.success(f"Denoising complete ({len(denoised_files)} files)")

    # --------------------
    # Mandatory sleep
    # --------------------
    st.info("Waiting 10 seconds before Stage 2...")
    time.sleep(10)

    # --------------------
    # Stage 2: Demucs
    # --------------------
    with st.spinner("Stage 2: Voice isolation (Demucs)..."):
        final_files, errors = process_directory_of_denoised_files(
            input_dir=stage1_dir,
            output_dir=stage2_dir,
            demucs_temp_out=DEMUCS_TEMP
        )

    st.success(f"Voice isolation complete ({len(final_files)} files)")

    if errors:
        st.warning("Some files failed:")
        for fname, err in errors:
            st.write(f"- {fname}: {err}")

    st.write(f"Final outputs stored in `{stage2_dir}`")


    # --------------------
    # Stage 3: ASR
    # --------------------
    st.info("Waiting 10 seconds before transcription...")
    time.sleep(10)

    with st.spinner("Stage 3: Transcribing isolated vocals..."):
        # Load ASR model once
        @st.cache_resource
        def get_asr():
            return load_asr_model()

        asr_model = get_asr()

        transcript_dir = TRANSCRIPT_ROOT / f"job_{job_id}"

        consolidated_csv, n_files = transcribe_final_outputs(
            input_dir=stage2_dir,
            output_csv_dir=transcript_dir,
            temp_root=ASR_TEMP,
            asr_model=asr_model
        )

    if consolidated_csv:
        st.success(f"Transcription complete ({n_files} files)")
        st.write(f"Saved to `{transcript_dir}`")
    else:
        st.warning("No final audio files were transcribed.")




    # --------------------
    # Stage 4: Irrelevance filtering
    # --------------------
    st.info("Waiting 10 seconds before relevance filtering...")
    time.sleep(10)

    @st.cache_resource
    def load_irr_models():
        return load_models(MODELS_ROOT)

    models, tokenizers, thresholds = load_irr_models()

    consolidated_csv = transcript_dir / "ALL_TRANSCRIPTS_CONSOLIDATED.csv"
    filtered_csv = transcript_dir / "ALL_TRANSCRIPTS_CONSOLIDATED_irr.csv"

    with st.spinner("Stage 4: Filtering irrelevant transcript rows..."):
        annotate_consolidated_csv(
            consolidated_csv=consolidated_csv,
            output_csv=filtered_csv,
            models=models,
            tokenizers=tokenizers,
            thresholds=thresholds,
        )

    st.success("Irrelevance filtering complete")
    st.write(f"Saved → `{filtered_csv}`")

    st.info("Waiting 10 seconds before sentiment analysis...")
    time.sleep(10)

    @st.cache_resource
    def get_sentiment_pipeline():
        return load_sentiment_pipeline()

    sentiment_clf = get_sentiment_pipeline()

    irr_csv = transcript_dir / "ALL_TRANSCRIPTS_CONSOLIDATED_irr.csv"
    sentiment_csv = transcript_dir / "SENTIMENT_CONSOLIDATED.csv"

    with st.spinner("Stage 5: Running sentiment analysis on relevant rows..."):
        out_csv, n_rows = run_sentiment_on_consolidated_csv(
            input_csv=irr_csv,
            output_csv=sentiment_csv,
            sentiment_pipeline=sentiment_clf,
        )

    st.success(f"Sentiment analysis complete ({n_rows} rows)")
    st.write(f"Saved → `{out_csv}`")

    st.session_state["job_id"] = job_id
    st.session_state["transcript_dir"] = str(transcript_dir)
    st.session_state["job_complete"] = True



# --------------------
# Config
# --------------------

if st.session_state.get("job_complete", False):
    transcript_dir = Path(st.session_state["transcript_dir"])
    SENTIMENT_CSV = transcript_dir / "SENTIMENT_CONSOLIDATED.csv"
    SENTIMENT_THRESHOLDS = [
        (0, 20,  "very_bad", "#8B0000"),
        (20, 50, "bad",      "#FF4C4C"),
        (50, 60, "neutral",  "#FFA500"),
        (60, 70, "okay",     "#FFD700"),
        (70, 85, "good",     "#9ACD32"),
        (85, 100,"excellent","#2ECC71"),
    ]


    def get_visuals(positive_pct):
        for lo, hi, emoji, color in SENTIMENT_THRESHOLDS:
            if lo <= positive_pct < hi:
                return emoji, color
        return "neutral", "#CCCCCC"


    # --------------------
    # Load sentiment results
    # --------------------
    st.header("Overall Sentiment Summary")

    if not SENTIMENT_CSV.exists():
        st.warning("Sentiment file not found.")
        st.stop()

    df = pd.read_csv(SENTIMENT_CSV)

    pos_count = (df["sentiment"] == "positive").sum()
    neg_count = (df["sentiment"] == "negative").sum()
    total = len(df)

    if total == 0:
        st.warning("No sentiment rows to display.")
        st.stop()

    positive_pct = round((pos_count / total) * 100, 2)

    emoji_name, bar_color = get_visuals(positive_pct)
    emoji_path = EMOJI_DIR / f"{emoji_name}.png"

    # --------------------
    # Layout
    # --------------------
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Positive Sentiment")

        st.markdown(
            f"""
            <div style="background-color:#eee;border-radius:10px;height:30px;">
                <div style="
                    width:{positive_pct}%;
                    background-color:{bar_color};
                    height:30px;
                    border-radius:10px;
                    text-align:center;
                    color:black;
                    font-weight:bold;">
                    {positive_pct}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            **Positive:** {pos_count}
            **Negative:** {neg_count}
            **Total:** {total}
            """
        )

    with col2:
        if emoji_path.exists():
            st.image(str(emoji_path), width=120)
        else:
            st.warning(f"Emoji not found: {emoji_name}")



    st.header("Sentiment Timeline")

    # Optional filters
    col1, col2 = st.columns(2)
    with col1:
        show_positive = st.checkbox("Show Positive", True)
    with col2:
        show_negative = st.checkbox("Show Negative", True)

    filtered_df = df.copy()

    if not show_positive:
        filtered_df = filtered_df[filtered_df["sentiment"] != "positive"]
    if not show_negative:
        filtered_df = filtered_df[filtered_df["sentiment"] != "negative"]

    # Sort by time if available
    if {"start", "end"}.issubset(filtered_df.columns):
        filtered_df = filtered_df.sort_values("start")

    # Scrollable container
    st.markdown(
        """
        <div style="
            max-height: 500px;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 10px;
        ">
        """,
        unsafe_allow_html=True,
    )

    # Render each row
    for _, row in filtered_df.iterrows():
        sentiment = row["sentiment"]
        text = row["transcription"]

        start = row.get("start", "")
        end = row.get("end", "")

        color = "#2ECC71" if sentiment == "positive" else "#E74C3C"

        st.markdown(
            f"""
            <div style="
                margin-bottom: 10px;
                padding: 8px;
                background-color: rgba(0,0,0,0.02);
                border-left: 6px solid {color};
                border-radius: 6px;
            ">
                <div style="font-size: 12px; color: #666;">
                    [{start} – {end}]
                </div>
                <div style="color: {color}; font-size: 16px;">
                    {text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Upload WAV files and run processing to view results.")
