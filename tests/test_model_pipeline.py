"""Tests unitaires de model_pipeline.py (données synthétiques, exécution < 10 s)."""

import numpy as np
import pandas as pd
import pytest

from model_pipeline import (
    categorize_diagnosis,
    clean_and_engineer,
    evaluate_model,
    load_model,
    save_model,
    train_model,
)


def make_raw_frame(n_rows: int = 80, seed: int = 0) -> pd.DataFrame:
    """Construit un petit DataFrame brut qui imite diabetic_data.csv."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "encounter_id": np.arange(n_rows),
            "patient_nbr": np.arange(n_rows) + 1000,
            "race": rng.choice(["Caucasian", "AfricanAmerican", "?"], n_rows),
            "gender": rng.choice(["Female", "Male", "Unknown/Invalid"], n_rows),
            "age": rng.choice(["[40-50)", "[60-70)", "[80-90)"], n_rows),
            "weight": "?",
            "admission_type_id": rng.integers(1, 4, n_rows),
            "discharge_disposition_id": rng.integers(1, 4, n_rows),
            "admission_source_id": rng.integers(1, 4, n_rows),
            "time_in_hospital": rng.integers(1, 14, n_rows),
            "payer_code": rng.choice(["MC", "?"], n_rows),
            "medical_specialty": rng.choice(["Cardiology", "?"], n_rows),
            "num_medications": rng.integers(1, 40, n_rows),
            "diag_1": rng.choice(["250.83", "428", "V57", "?"], n_rows),
            "diag_2": rng.choice(["250", "786", "E888", "?"], n_rows),
            "diag_3": rng.choice(["401", "585", "?"], n_rows),
            "max_glu_serum": rng.choice([None, "Norm", ">200"], n_rows),
            "A1Cresult": rng.choice([None, "Norm", ">8"], n_rows),
            "change": rng.choice(["No", "Ch"], n_rows),
            "diabetesMed": rng.choice(["No", "Yes"], n_rows),
        }
    )


@pytest.mark.parametrize(
    "code, expected",
    [
        ("250.83", "Endocrine_Metabolic"),
        ("428", "Circulatory"),
        ("786", "Symptoms_Signs"),
        ("V57", "External_Cause"),
        ("E888", "External_Cause"),
        ("Unknown", "Unknown"),
    ],
)
def test_categorize_diagnosis(code, expected):
    assert categorize_diagnosis(code) == expected


def test_clean_and_engineer_removes_missing_and_text_codes():
    cleaned = clean_and_engineer(make_raw_frame())

    for col in ("weight", "encounter_id", "patient_nbr"):
        assert col not in cleaned.columns
    assert not (cleaned == "?").any().any()
    assert cleaned["max_glu_serum"].notna().all()
    assert set(cleaned["gender"].unique()) <= {0, 1, 2}
    assert set(cleaned["change"].unique()) <= {0, 1}
    assert cleaned["age"].between(0, 9).all()


def test_train_save_load_roundtrip(tmp_path):
    X = clean_and_engineer(make_raw_frame(n_rows=120, seed=1))
    y = pd.Series(np.random.default_rng(2).integers(0, 2, len(X)))

    model = train_model(X, y, "decision_tree")
    path = save_model(model, str(tmp_path / "models" / "dt.joblib"))
    reloaded = load_model(path)

    assert (model.predict(X) == reloaded.predict(X)).all()
    metrics = evaluate_model(reloaded, X, y, "dt", verbose=False)
    assert 0.0 <= metrics["Accuracy"] <= 1.0


def test_load_model_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(str(tmp_path / "absent.joblib"))


def test_train_model_unknown_name_raises():
    X = clean_and_engineer(make_raw_frame(n_rows=30))
    y = pd.Series([0, 1] * 15)
    with pytest.raises(ValueError):
        train_model(X, y, "modele_inexistant")
