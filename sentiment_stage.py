from pathlib import Path
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)


def load_sentiment_pipeline():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    device = 0 if torch.cuda.is_available() else -1

    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=False,
        device=device,
    )

    return clf


def run_sentiment_on_consolidated_csv(
    input_csv: Path,
    output_csv: Path,
    sentiment_pipeline,
):
    df = pd.read_csv(input_csv)

    required_cols = {"transcription", "Irrelevent"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns: {required_cols}"
        )

    # 1. Keep only relevant rows
    df = df[df["Irrelevent"] != 1].copy()

    if df.empty:
        df.to_csv(output_csv, index=False)
        return output_csv, 0

    # 2. Sentiment classification
    df["sentiment"] = df["transcription"].astype(str).apply(
        lambda x: sentiment_pipeline(x)[0]["label"]
    )

    # 3. Drop neutral
    df = df[df["sentiment"] != "neutral"]

    # 4. Track source
    df["source_file"] = input_csv.name

    df.to_csv(output_csv, index=False)
    return output_csv, len(df)

