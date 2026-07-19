"""
MedAssist AI — symptom-to-disease classifier training script.

Run once (or whenever ml/data/*.csv changes) to regenerate the .pkl
artifacts predict.py loads at inference time:

    python train_model.py

Data:
  - Symptom2Disease.csv: 1200 free-text symptom descriptions across 24
    diseases — the primary training set, and the one accuracy is measured
    against (matches the app's actual input: a free-text textarea, not a
    checkbox list). Its Migraine rows were originally mislabeled (describing
    GERD/vision/mood symptoms rather than headaches), which made real
    migraine descriptions misclassify as Malaria; they've since been
    replaced with real classic-presentation text (throbbing/one-sided
    headache, photophobia, phonophobia, nausea, aura).
  - correct_symptoms.csv: a small supplementary set covering conditions not
    present in Symptom2Disease.csv. Classes with fewer than 5 examples were
    dropped entirely (Oral Thrush, Aphthous Ulcer, Cold Sore, Gastritis,
    Epistaxis, Gastroenteritis) — a class trained on 1-2 examples doesn't
    learn anything, it just adds noise that can hijack unrelated
    predictions. Only "Oral Ulcer (Canker Sore)" (6 examples) remains, and
    is folded into training only, same as before: still too few examples to
    carve out its own held-out split.

Model: TF-IDF (unigrams + bigrams) -> LogisticRegression, wrapped in
CalibratedClassifierCV so predict_proba is actually calibrated (plain
LogisticRegression probabilities look calibrated but aren't, once classes
overlap as heavily as they do here — the app reports predict_proba directly
as the user-facing confidence %, so an uncalibrated score means confidently
wrong answers).

Accuracy is measured with stratified k-fold CV on Symptom2Disease.csv alone,
refitting the vectorizer inside each fold (no vocabulary leakage from the
held-out fold into TF-IDF's IDF weights). A single 80/20 split was dropped
because this dataset contains many near-duplicate templated paraphrases
(e.g. two Malaria rows are byte-for-byte identical) — one split could land
lucky or unlucky; k-fold reports a mean and spread instead of one point
estimate that's easy to over-read.
"""

import json

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent

RANDOM_STATE = 42
N_FOLDS = 5


def load_data():
    primary = pd.read_csv(DATA_DIR / "Symptom2Disease.csv")
    primary = primary[["label", "text"]].dropna()
    # Labels are inconsistently cased in the source data ("allergy" vs
    # "Malaria") — leave them as-is rather than normalizing, since
    # doctor_map.json's keys were written to match these exact strings.

    supplementary = pd.read_csv(DATA_DIR / "correct_symptoms.csv")
    supplementary = supplementary[["label", "text"]].dropna()

    return primary, supplementary


def build_vectorizer():
    return TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=5000,
        sublinear_tf=True,
    )


def cross_validate(primary: pd.DataFrame) -> None:
    """Stratified k-fold CV on the primary set, printed as an honest
    generalization estimate (not the number that ships)."""
    texts = primary["text"].to_numpy()
    labels = primary["label"].to_numpy()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_accuracies = []
    all_true, all_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, labels), start=1):
        vectorizer = build_vectorizer()
        X_train = vectorizer.fit_transform(texts[train_idx])
        X_test = vectorizer.transform(texts[test_idx])

        model = LogisticRegression(max_iter=2000, class_weight="balanced")
        model.fit(X_train, labels[train_idx])

        preds = model.predict(X_test)
        acc = accuracy_score(labels[test_idx], preds)
        fold_accuracies.append(acc)
        all_true.extend(labels[test_idx])
        all_pred.extend(preds)
        print(f"  Fold {fold}/{N_FOLDS} accuracy: {acc:.2%}")

    fold_accuracies = np.array(fold_accuracies)
    print(
        f"\n{N_FOLDS}-fold CV accuracy on Symptom2Disease.csv: "
        f"{fold_accuracies.mean():.2%} (+/- {fold_accuracies.std():.2%})"
    )
    print(classification_report(all_true, all_pred, zero_division=0))


def main():
    primary, supplementary = load_data()

    print(f"Cross-validating on {len(primary)} primary rows ({N_FOLDS}-fold)...")
    cross_validate(primary)

    # Final artifact ships on every available row (primary + supplementary)
    # — the CV above exists only to measure generalization, not to withhold
    # data from the deployed model. Wrapped in CalibratedClassifierCV so the
    # confidence % predict.py reports is trustworthy rather than just
    # LogisticRegression's raw (uncalibrated, prone-to-overconfidence) score.
    full_df = pd.concat([primary, supplementary], ignore_index=True)
    final_vectorizer = build_vectorizer()
    X_full = final_vectorizer.fit_transform(full_df["text"])

    base_model = LogisticRegression(max_iter=2000, class_weight="balanced")
    calibrated_model = CalibratedClassifierCV(base_model, cv=5, method="isotonic")
    calibrated_model.fit(X_full, full_df["label"])

    joblib.dump(calibrated_model, OUT_DIR / "disease_model.pkl")
    joblib.dump(final_vectorizer, OUT_DIR / "vectorizer.pkl")
    print(f"\nSaved disease_model.pkl and vectorizer.pkl to {OUT_DIR}")
    print(f"Classes ({len(calibrated_model.classes_)}): {sorted(calibrated_model.classes_)}")

    # Sanity-check every trained class has a doctor_map.json entry — a
    # missing one silently falls back to General Physician at inference
    # time, which is a real behavior change worth catching here instead.
    with open(OUT_DIR / "doctor_map.json") as f:
        doctor_map = json.load(f)
    unmapped = [c for c in calibrated_model.classes_ if c not in doctor_map]
    if unmapped:
        print(f"\nWARNING: classes with no doctor_map.json entry (will default to General Physician): {unmapped}")


if __name__ == "__main__":
    main()
