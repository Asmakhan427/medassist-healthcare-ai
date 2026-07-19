"""
MedAssist AI — symptom prediction CLI bridge.

Invoked by packages/backend/src/services/python.service.ts as:
    python predict.py "<free-text symptom description>"

Prints a single JSON object on stdout matching PythonPredictionResult:
    { disease, confidence, severity, emergency, doctor, recommendations }

Loads disease_model.pkl + vectorizer.pkl (produced by train_model.py) once
per invocation — see python.service.ts for why this is a short-lived
process per request rather than a long-running server: it mirrors the
original server.js behavior and keeps the Node/Python boundary simple, at
the cost of paying model-load time on every call. Fine for this app's
volume; revisit with a persistent Python process (e.g. a small Flask/FastAPI
sidecar) if that ever becomes a bottleneck.
"""

import json
import re
import sys
from pathlib import Path

import joblib
import pandas as pd

ML_DIR = Path(__file__).parent

# Curated, safety-critical phrases that force CRITICAL/emergency regardless
# of the statistical severity score below — the kind of thing that should
# never depend solely on a small text classifier's confidence.
EMERGENCY_PHRASES = [
    "chest pain",
    "pain in chest",
    "chest hurts",
    "pain in my chest",
    "heart attack",
    "can't breathe",
    "cant breathe",
    "cannot breathe",
    "difficulty breathing",
    "trouble breathing",
    "hard to breathe",
    "struggling to breathe",
    "gasping for air",
    "not breathing",
    "severe bleeding",
    "heavy bleeding",
    "unconscious",
    "unresponsive",
    "stroke",
    "seizure",
    "suicidal",
    "severe allergic reaction",
    "anaphylaxis",
    "coughing blood",
    "blue lips",
]

SEVERITY_THRESHOLDS = [
    (22, "CRITICAL"),
    (15, "SEVERE"),
    (7, "MODERATE"),
]

# Below this, the top prediction isn't trustworthy enough to state as a
# diagnosis-shaped answer (see determine_severity's caller in main()) —
# with overlapping classes and a small training set, a wrong prediction can
# still land at 60-80% raw probability, so "top score, whatever it is" is
# not a safe thing to show a user in a medical app. 0.4 is a starting point,
# not tuned against a held-out validation set; revisit once real usage data
# exists to pick a threshold that trades off abstention rate vs error rate.
CONFIDENCE_ABSTAIN_THRESHOLD = 0.45
UNCERTAIN_DIAGNOSIS = "Uncertain — please consult a doctor for evaluation"

# Common phrasings that don't literally contain the Symptom-severity.csv
# key they describe, so the original substring match silently missed them
# (e.g. "sore throat" never matches "patches_in_throat" or
# "throat_irritation"; "my head aches" never matches "headache"). Mapped to
# the SAME weight as their canonical entry via build_symptom_patterns()
# below, and matched with a word-boundary regex OR'd across all variants so
# a single mention of the concept is scored once — not missed, and not
# double-counted if the user happens to use two synonyms at once.
SEVERITY_SYNONYMS: dict[str, list[str]] = {
    "headache": ["head ache", "head aches", "head hurts", "head hurting", "head pain"],
    "high fever": ["high temperature", "very high fever"],
    "mild fever": ["low grade fever", "slight fever", "low fever"],
    "stomach pain": ["stomach ache", "stomachache", "tummy pain", "belly ache"],
    "abdominal pain": ["tummy ache"],
    "throat irritation": ["sore throat", "scratchy throat"],
    "vomiting": ["throwing up", "vomited"],
    "nausea": ["nauseous", "feel sick", "feeling sick", "queasy"],
    "cough": ["coughing"],
    "breathlessness": ["shortness of breath", "short of breath", "out of breath"],
    "chest pain": ["pain in chest", "chest hurts"],
    "back pain": ["backache"],
    "neck pain": ["sore neck", "stiff neck ache"],
    "joint pain": ["achy joints", "joints ache", "joints hurt"],
    "muscle pain": ["muscle ache", "body ache", "body aches"],
    "dizziness": ["dizzy", "light headed", "lightheaded"],
    "fatigue": ["exhausted", "worn out", "no energy"],
    "runny nose": ["nose is running", "runny nostrils"],
    "diarrhoea": ["diarrhea", "loose motions", "loose stools"],
    "loss of appetite": ["not hungry", "no appetite"],
    "weakness in limbs": ["weak limbs", "limbs feel weak"],
}


def load_severity_weights():
    df = pd.read_csv(ML_DIR / "data" / "Symptom-severity.csv")
    df.columns = [c.strip() for c in df.columns]
    weights = {}
    for _, row in df.iterrows():
        symptom = str(row["Symptom"]).strip().lower()
        if not symptom or symptom == "nan":
            continue
        # "skin_rash" -> "skin rash", so it can be substring-matched
        # against free-text input the way a user would actually phrase it.
        phrase = symptom.replace("_", " ")
        weights[phrase] = int(row["weight"])
    return weights


def build_symptom_patterns(severity_weights: dict) -> dict:
    """canonical phrase -> compiled regex matching that phrase or any of
    its known synonyms, word-boundary delimited so "ear ache" doesn't
    match inside an unrelated longer word."""
    patterns = {}
    for phrase in severity_weights:
        variants = [phrase] + SEVERITY_SYNONYMS.get(phrase, [])
        alternation = "|".join(re.escape(v) for v in variants)
        patterns[phrase] = re.compile(rf"\b(?:{alternation})\b")
    return patterns


def load_doctor_map():
    with open(ML_DIR / "doctor_map.json") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def load_precautions():
    df = pd.read_csv(ML_DIR / "data" / "symptom_precaution.csv")
    lookup = {}
    for _, row in df.iterrows():
        key = str(row["Disease"]).strip().lower()
        precautions = [
            str(row[c]).strip()
            for c in df.columns[1:]
            if pd.notna(row[c]) and str(row[c]).strip()
        ]
        lookup[key] = precautions

    # symptom_precaution.csv uses different casing/spelling than the
    # training labels for several diseases (case differences are handled by
    # the lowercase lookup below; these need an explicit alias).
    aliases = {
        "dimorphic hemorrhoids": "dimorphic hemmorhoids(piles)",
        "gastroesophageal reflux disease": "gerd",
        "peptic ulcer disease": "peptic ulcer diseae",  # matches source typo
    }
    for trained_label, precaution_key in aliases.items():
        if precaution_key in lookup:
            lookup[trained_label] = lookup[precaution_key]

    return lookup


def determine_severity(
    symptoms_lower: str, severity_weights: dict, symptom_patterns: dict
) -> tuple[str, bool]:
    if any(phrase in symptoms_lower for phrase in EMERGENCY_PHRASES):
        return "CRITICAL", True

    score = sum(
        weight
        for phrase, weight in severity_weights.items()
        if symptom_patterns[phrase].search(symptoms_lower)
    )

    # Bare "fever" (no severity qualifier) isn't itself a Symptom-severity.csv
    # key -- only "high fever" / "mild fever" are -- so without this an
    # unqualified "I have a fever" scores 0 for it entirely. Only add it when
    # neither qualified form already matched, so a mention of "high fever"
    # doesn't also get scored again for the bare word "fever" it contains.
    already_scored_fever = symptom_patterns["high fever"].search(
        symptoms_lower
    ) or symptom_patterns["mild fever"].search(symptoms_lower)
    if not already_scored_fever and re.search(r"\bfevers?\b", symptoms_lower):
        score += severity_weights["mild fever"]

    for threshold, level in SEVERITY_THRESHOLDS:
        if score >= threshold:
            return level, level == "CRITICAL"
    return "MILD", False


def get_recommendations(disease: str, precautions: dict) -> str:
    matched = precautions.get(disease.strip().lower())
    if matched:
        return "Recommended precautions: " + "; ".join(matched) + "."
    return "Consult a doctor for a full evaluation and personalized treatment plan."


def main():
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print(json.dumps({"error": "No symptoms provided"}), file=sys.stderr)
        sys.exit(1)

    symptoms = sys.argv[1]
    symptoms_lower = re.sub(r"\s+", " ", symptoms.lower()).strip()

    model = joblib.load(ML_DIR / "disease_model.pkl")
    vectorizer = joblib.load(ML_DIR / "vectorizer.pkl")
    doctor_map = load_doctor_map()
    severity_weights = load_severity_weights()
    symptom_patterns = build_symptom_patterns(severity_weights)
    precautions = load_precautions()

    features = vectorizer.transform([symptoms])
    probabilities = model.predict_proba(features)[0]
    best_idx = probabilities.argmax()
    disease = model.classes_[best_idx]
    confidence = round(float(probabilities[best_idx]) * 100, 1)

    severity, emergency = determine_severity(symptoms_lower, severity_weights, symptom_patterns)

    # Below CONFIDENCE_ABSTAIN_THRESHOLD, don't present the top class as if
    # it were a diagnosis -- with overlapping classes and a small training
    # set, a wrong prediction can still land at 60-80% raw probability. The
    # true confidence is still reported so the caller can show it, but the
    # disease label and doctor recommendation route to a safe default
    # instead of a specific (possibly wrong) condition. Emergency detection
    # above is unaffected -- it comes from the raw text, not this model.
    if probabilities[best_idx] < CONFIDENCE_ABSTAIN_THRESHOLD:
        doctor = "General Physician"
        recommendations = get_recommendations(UNCERTAIN_DIAGNOSIS, precautions)
        disease = UNCERTAIN_DIAGNOSIS
    else:
        doctor = doctor_map.get(disease, "General Physician")
        recommendations = get_recommendations(disease, precautions)

    result = {
        "disease": str(disease),
        "confidence": confidence,
        "severity": severity,
        "emergency": emergency,
        "doctor": doctor,
        "recommendations": recommendations,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
