from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


LABEL_COLS = ["Irrelevent"]


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models(models_root: Path):
    """
    Loads all irrelevance classifiers and thresholds.
    """
    device = get_device()

    models = {}
    tokenizers = {}
    thresholds = {}

    for label in LABEL_COLS:
        model_dir = models_root / label
        if not model_dir.is_dir():
            continue

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(device)
        model.eval()

        thr_path = model_dir / "threshold.json"
        if thr_path.exists():
            with open(thr_path, "r") as f:
                thr = float(json.load(f).get("best_threshold", 0.5))
        else:
            thr = 0.5

        tokenizers[label] = tokenizer
        models[label] = model
        thresholds[label] = thr

    return models, tokenizers, thresholds


def predict_probs(
    texts,
    tokenizer,
    model,
    batch_size=32,
    max_length=128,
):
    """
    Returns np.array of P(class=1) for each text.
    """
    device = next(model.parameters()).device
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        enc = tokenizer(
            batch,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_probs)


def annotate_consolidated_csv(
    consolidated_csv: Path,
    output_csv: Path,
    models,
    tokenizers,
    thresholds,
):
    df = pd.read_csv(consolidated_csv)

    if "transcription" not in df.columns:
        raise ValueError("Expected 'transcription' column not found")

    texts = df["transcription"].astype(str).tolist()

    for label in LABEL_COLS:
        if label not in models:
            continue

        probs = predict_probs(
            texts,
            tokenizers[label],
            models[label],
        )

        thr = thresholds.get(label, 0.5)
        preds = (probs >= thr).astype(int)
        conf = np.where(preds == 1, probs, 1.0 - probs)

        df[label] = preds
        df[f"conf_{label}"] = np.round(conf, 2)

    df.to_csv(output_csv, index=False)
    return output_csv

