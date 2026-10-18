"""
STEP 7 — Evaluation, GridSearchCV + SMOTE pipeline model
=============================================================
Identical evaluation logic to step7_evaluation_relaxed.py, pointed at
the tuned pipeline model from step6_random_forest_gridsearch.py
(random_forest_model_gridsearch.joblib) instead.

Note: the saved model here is a full imblearn Pipeline ([SMOTE ->
RandomForest]), not a bare RandomForestClassifier - but .predict()
works identically either way, since SMOTE automatically does nothing
at prediction time (it only ever acts during .fit()). No other code
below needs to change because of this.

Two layers of evaluation:

  1. BEAT-LEVEL: the standard per-AAMI-class recall/precision/
     specificity, computed on the model's raw beat-by-beat predictions
     against the untouched test set.

  2. PATIENT-LEVEL: the model's beat predictions for each test patient
     are aggregated into ONE Normal/Arrhythmic label per patient, using
     the EXACT SAME burden rule from Step 3b (V burden > 10% or S
     burden > 10% -> Arrhythmic). That predicted label is then compared
     against the TRUE patient label Step 3b computed from the real
     annotations, to see whether the model's beat-level mistakes wash
     out or compound once aggregated up to the level that actually
     matters for this project - is the PATIENT arrhythmic.

IMPORTANT CAVEAT carried into this step: the predicted burden for each
patient is computed only from the beats that survived Steps 4-5 (the
relaxed retention + imputation pipeline), not that patient's full
original beat count. A patient's true burden (Step 3b) was computed
from ALL their real beats; their predicted burden here is computed from
a partial, pipeline-filtered subset. This is the same bias we've been
tracking since Step 4 - worth restating here since it's the last stage
where it can still affect a result.

Also worth restating: with only 22 test patients, patient-level metrics
are statistically fragile (misclassifying 1-2 patients swings the
result by several percentage points) - report these with appropriate
caution, not as if they were computed from a large sample.
"""

import numpy as np
import pandas as pd
import joblib
import smote_utils  # noqa: F401 - must be imported so joblib can find build_sampling_strategy when unpickling the model

CLASSES = ["N", "S", "V", "F", "Q"]
RECALL_THRESHOLD = 0.85
PRECISION_THRESHOLD = 0.80
SPECIFICITY_THRESHOLD = 0.80

# Same thresholds as Step 3b - the whole point is using an IDENTICAL rule
# on predicted beats that was used on the true annotations, so the two
# labels are actually comparable.
V_THRESHOLD_PCT = 10.0
S_THRESHOLD_PCT = 10.0

FEATURE_COLUMNS = [
    "pre_rr", "post_rr", "local_avg_rr",
    "qrs_duration", "pr_interval", "qt_interval", "p_amplitude",
    "skewness", "kurtosis",
]


def evaluate_beat_level(y_test, y_pred) -> pd.DataFrame:
    """
    Same per-class recall/precision/specificity approach used in the
    strict pipeline's evaluation, computed here on the relaxed test set.
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred, labels=CLASSES)
    total = cm.sum()

    rows = []
    for i, cls in enumerate(CLASSES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp

        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

        rows.append({
            "class": cls, "support": tp + fn,
            "recall": recall, "precision": precision, "specificity": specificity,
        })

    return pd.DataFrame(rows), cm


def aggregate_to_patient_predictions(test_df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Takes the model's beat-level PREDICTIONS and rolls them up into one
    predicted Normal/Arrhythmic label per patient, using the identical
    threshold rule from Step 3b - just applied to what the model
    guessed, instead of the real annotations.
    """
    df = test_df[["record"]].copy()
    df["predicted_beat_label"] = y_pred

    rows = []
    for record, group in df.groupby("record"):
        total = len(group)
        counts = group["predicted_beat_label"].value_counts()

        v_pct = 100 * counts.get("V", 0) / total
        s_pct = 100 * counts.get("S", 0) / total

        predicted_label = "Arrhythmic" if (v_pct > V_THRESHOLD_PCT or s_pct > S_THRESHOLD_PCT) else "Normal"

        rows.append({
            "record": record, "beats_evaluated": total,
            "predicted_V_pct": v_pct, "predicted_S_pct": s_pct,
            "predicted_patient_label": predicted_label,
        })

    return pd.DataFrame(rows).sort_values("record").reset_index(drop=True)


def evaluate_patient_level(predicted_df: pd.DataFrame, true_labels_path: str = "patient_labels.csv") -> pd.DataFrame:
    """
    Compares the predicted patient labels against the TRUE patient
    labels Step 3b computed from the real annotations, for just the 22
    test-set patients. Returns a merged table with both labels side by
    side, plus a "correct" flag, ready for a simple binary
    Normal-vs-Arrhythmic confusion matrix.
    """
    true_df = pd.read_csv(true_labels_path)
    true_df["record"] = true_df["record"].astype(str)

    merged = predicted_df.merge(
        true_df[["record", "patient_label"]].rename(columns={"patient_label": "true_patient_label"}),
        on="record", how="left",
    )
    merged["correct"] = merged["true_patient_label"] == merged["predicted_patient_label"]

    return merged


def print_patient_confusion(merged: pd.DataFrame) -> None:
    """
    Prints a simple 2x2 (Normal vs Arrhythmic) confusion matrix and its
    recall/precision/specificity, treating "Arrhythmic" as the positive
    class (the one we actually care about detecting).
    """
    tp = ((merged["true_patient_label"] == "Arrhythmic") & (merged["predicted_patient_label"] == "Arrhythmic")).sum()
    fn = ((merged["true_patient_label"] == "Arrhythmic") & (merged["predicted_patient_label"] == "Normal")).sum()
    fp = ((merged["true_patient_label"] == "Normal") & (merged["predicted_patient_label"] == "Arrhythmic")).sum()
    tn = ((merged["true_patient_label"] == "Normal") & (merged["predicted_patient_label"] == "Normal")).sum()

    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    print(f"\nPatient-level confusion matrix (n = {len(merged)} test patients):")
    print(f"                    Predicted Arrhythmic   Predicted Normal")
    print(f"True Arrhythmic     {tp:>19}   {fn:>16}")
    print(f"True Normal         {fp:>19}   {tn:>16}")

    print(f"\nPatient-level Recall:      {recall:.3f}  (of truly arrhythmic patients, caught)")
    print(f"Patient-level Precision:   {precision:.3f}  (of patients flagged arrhythmic, correct)")
    print(f"Patient-level Specificity: {specificity:.3f}  (of truly normal patients, correctly left alone)")
    print("\nCAUTION: these are computed from only", len(merged), "patients - "
          "1-2 misclassified patients will swing these numbers substantially.")


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    # Loading the tuned [SMOTE -> RandomForest] PIPELINE from the grid
    # search. test_audit_relaxed.csv is the same untouched test set
    # used by every other version in this project.
    model = joblib.load("random_forest_model_gridsearch.joblib")
    test_df = pd.read_csv("test_audit_relaxed.csv")  # has record + label + the 9 features together
    test_df["record"] = test_df["record"].astype(str)  # keep dtype consistent with patient_labels.csv's "record" column

    y_test = test_df["label"]
    y_pred = model.predict(test_df[FEATURE_COLUMNS])

    # --- Layer 1: beat-level evaluation ---
    print("=" * 70)
    print("BEAT-LEVEL EVALUATION")
    print("=" * 70)
    beat_metrics, cm = evaluate_beat_level(y_test, y_pred)
    print(beat_metrics.round(3).to_string(index=False))

    macro_recall = beat_metrics["recall"].mean()
    macro_precision = beat_metrics["precision"].mean()
    macro_specificity = beat_metrics["specificity"].mean()
    print(f"\nMacro-averaged Recall:      {macro_recall:.3f} "
          f"({'PASS' if macro_recall >= RECALL_THRESHOLD else 'FAIL'} threshold {RECALL_THRESHOLD})")
    print(f"Macro-averaged Precision:   {macro_precision:.3f} "
          f"({'PASS' if macro_precision >= PRECISION_THRESHOLD else 'FAIL'} threshold {PRECISION_THRESHOLD})")
    print(f"Macro-averaged Specificity: {macro_specificity:.3f} "
          f"({'PASS' if macro_specificity >= SPECIFICITY_THRESHOLD else 'FAIL'} threshold {SPECIFICITY_THRESHOLD})")

    # --- Layer 2: aggregate predictions up to patient level ---
    print("\n" + "=" * 70)
    print("PATIENT-LEVEL EVALUATION")
    print("=" * 70)
    predicted_df = aggregate_to_patient_predictions(test_df, y_pred)
    merged = evaluate_patient_level(predicted_df)

    print("\nPer-patient comparison (true vs. predicted):")
    print(merged[["record", "beats_evaluated", "predicted_V_pct", "predicted_S_pct",
                   "true_patient_label", "predicted_patient_label", "correct"]].round(2).to_string(index=False))

    n_correct = merged["correct"].sum()
    print(f"\nOverall patient-level accuracy: {n_correct}/{len(merged)} "
          f"({100 * n_correct / len(merged):.1f}%)")

    print_patient_confusion(merged)

    beat_metrics.to_csv("beat_level_metrics_gridsearch.csv", index=False)
    merged.to_csv("patient_level_predictions_gridsearch.csv", index=False)
    print("\nSaved beat_level_metrics_gridsearch.csv, patient_level_predictions_gridsearch.csv")
    print("\n(Compare against the _relaxed.csv and _no_smote.csv versions from the earlier runs)")
