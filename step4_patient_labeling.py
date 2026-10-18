"""
STEP 3b — Patient-Level Labeling (Normal vs. Arrhythmic)
===========================================================
Goal: label each of the 48 MIT-BIH patients as "Normal" or "Arrhythmic"
based on how often arrhythmic beat types actually appear in their
30-minute recording - not just whether any arrhythmic beat ever
appears at all. This mirrors real clinical practice: an occasional
ectopic beat (e.g. in a fit, well-trained patient) doesn't make someone
"arrhythmic", but a high burden of ectopic beats does.

Runs on the OUTPUT OF STEP 3 (y_labels.npy, groups.npy) - i.e. on the
raw AAMI-mapped, post-segmentation beat counts - deliberately BEFORE
Step 4's feature extraction. We already found that feature extraction
disproportionately drops V/F/Q beats (63-70%) versus N (18%); labeling
patients from post-extraction beat counts would measure "how arrhythmic
is this patient" through a lens that has already destroyed most of the
arrhythmic evidence. Labeling here, on the untouched beat counts,
avoids that bias entirely.

Thresholds used (see methodology write-up for full literature citations):
  V (ventricular ectopic) burden > 10%  -> arrhythmic
      (Slaven et al., 2025; Yoneda & Stevenson, 2025)
  S (supraventricular ectopic) burden > 10% -> arrhythmic
      (ScienceDirect, 2018 - "frequent" SVE threshold)
  F (fusion) burden -> reported, but NOT used in the decision.
      No clinical burden literature exists for F; it is a morphological
      artifact of two depolarization wavefronts merging, not a distinct
      arrhythmic disease process, so no threshold could be justified.
  Q (unknown/paced) burden -> reported and flagged separately as a
      device-status marker, NOT folded into the arrhythmic decision.
      A patient dominated by paced beats isn't "arrhythmic" in the
      disease sense - they have a functioning pacemaker.
"""

import numpy as np
import pandas as pd

# --- thresholds (see docstring above for literature justification) ---
V_THRESHOLD_PCT = 10.0
S_THRESHOLD_PCT = 10.0
PACED_THRESHOLD_PCT = 50.0  # a patient whose beats are mostly Q is flagged "paced", not "arrhythmic"

CLASSES = ["N", "S", "V", "F", "Q"]


def compute_patient_burden(y_labels: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    """
    Computes, for every patient (record), how many beats of each AAMI
    class they have, and TWO versions of "what percentage is that":

      *_pct       -> percentage of ALL beats (including paced/Q beats)
      *_pct_nonQ  -> percentage of only the NON-paced beats

    Why both: for a heavily-paced patient, Q beats can dominate the
    total count and dilute every other class's percentage, masking a
    genuinely high ventricular or supraventricular burden AMONG the
    beats where the heart is actually driving itself (not the pacemaker).
    Reporting both lets us see whether a patient's underlying native
    rhythm looks different once paced beats are set aside.
    """
    df = pd.DataFrame({"record": groups, "label": y_labels})

    rows = []
    for record, group_df in df.groupby("record"):
        total = len(group_df)
        counts = group_df["label"].value_counts()
        q_count = int(counts.get("Q", 0))
        non_q_total = total - q_count

        row = {"record": record, "total_beats": total, "non_Q_beats": non_q_total}
        for cls in CLASSES:
            count = int(counts.get(cls, 0))
            row[f"{cls}_count"] = count
            row[f"{cls}_pct"] = 100 * count / total
            if cls != "Q":
                # guard against a fully-paced patient with zero non-Q beats
                row[f"{cls}_pct_nonQ"] = (100 * count / non_q_total) if non_q_total > 0 else np.nan
        rows.append(row)

    return pd.DataFrame(rows).sort_values("record").reset_index(drop=True)


def assign_patient_labels(burden_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the threshold rule TWICE - once using each class's share of
    ALL beats, once using each class's share of only NON-PACED beats -
    so both views of "how arrhythmic is this patient" are available
    side by side.

      patient_label       -> based on V_pct / S_pct (share of all beats)
      patient_label_nonQ   -> based on V_pct_nonQ / S_pct_nonQ (share of
                              native, non-paced beats only)
      label_differs        -> True where the two disagree - these are
                              exactly the patients where paced-beat
                              dilution changes the conclusion, worth
                              inspecting individually rather than
                              trusting either version blindly.

    paced_flag is unchanged: True if Q_pct > 50% of ALL beats.
    """
    burden_df = burden_df.copy()

    is_arrhythmic = (burden_df["V_pct"] > V_THRESHOLD_PCT) | (burden_df["S_pct"] > S_THRESHOLD_PCT)
    burden_df["patient_label"] = np.where(is_arrhythmic, "Arrhythmic", "Normal")

    is_arrhythmic_nonQ = (burden_df["V_pct_nonQ"] > V_THRESHOLD_PCT) | (burden_df["S_pct_nonQ"] > S_THRESHOLD_PCT)
    burden_df["patient_label_nonQ"] = np.where(is_arrhythmic_nonQ, "Arrhythmic", "Normal")
    # a patient with zero non-Q beats has an undefined non-Q label
    burden_df.loc[burden_df["non_Q_beats"] == 0, "patient_label_nonQ"] = "Undefined (no native beats)"

    burden_df["paced_flag"] = burden_df["Q_pct"] > PACED_THRESHOLD_PCT
    burden_df["label_differs"] = burden_df["patient_label"] != burden_df["patient_label_nonQ"]

    return burden_df


if __name__ == "__main__":
    y_labels = np.load("y_labels.npy", allow_pickle=True)
    groups = np.load("groups.npy", allow_pickle=True)

    burden_df = compute_patient_burden(y_labels, groups)
    burden_df = assign_patient_labels(burden_df)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    print("Per-patient AAMI class burden and resulting labels (both versions):\n")
    display_cols = ["record", "total_beats", "non_Q_beats",
                     "V_pct", "S_pct", "V_pct_nonQ", "S_pct_nonQ",
                     "patient_label", "patient_label_nonQ", "paced_flag", "label_differs"]
    print(burden_df[display_cols].round(2).to_string(index=False))

    print("\nPatient label counts (share of ALL beats):")
    print(burden_df["patient_label"].value_counts().to_string())

    print("\nPatient label counts (share of NON-PACED beats only):")
    print(burden_df["patient_label_nonQ"].value_counts().to_string())

    print("\nPatients where the two versions DISAGREE (paced-beat dilution changed the conclusion):")
    disagreements = burden_df.loc[burden_df["label_differs"],
                                   ["record", "V_pct", "V_pct_nonQ", "S_pct", "S_pct_nonQ",
                                    "patient_label", "patient_label_nonQ"]]
    print(disagreements.round(2).to_string(index=False) if len(disagreements) else "None")

    burden_df.to_csv("patient_labels.csv", index=False)
    print("\nSaved patient_labels.csv")
