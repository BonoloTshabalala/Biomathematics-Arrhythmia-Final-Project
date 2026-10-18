"""
STEP 5 (RELAXED) — Split First, Then Impute (Leakage-Safe)
==============================================================
Goal: apply the same fixed inter-patient train/test split as before,
but do it BEFORE filling in the qrs_duration/pr_interval/qt_interval
gaps Step 4 (relaxed) deliberately left as NaN - and compute the
fallback reference values used for imputation from TRAINING patients
only.

WHY THIS ORDER MATTERS (recap): if we imputed before splitting, a
patient with too few of their own same-class beats would get their
missing values filled using a median pooled across ALL 48 patients -
including future test patients. That's a data leak: information from
the test set would quietly shape values used to train the model. This
is exactly the same principle as fitting a scikit-learn scaler or
imputer only on training data, then applying that already-fitted
transform to the test set - preparation statistics come from training
data ONLY, then get applied to both sets.

Run step4_feature_extraction_relaxed.py first - this script reads its
output (ecg_features_relaxed_raw.csv), which still has real NaNs in it.
"""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "pre_rr", "post_rr", "local_avg_rr",
    "qrs_duration", "pr_interval", "qt_interval", "p_amplitude",
    "skewness", "kurtosis",
]
IMPUTABLE_FEATURES = ["qrs_duration", "pr_interval", "qt_interval"]
MIN_LOCAL_SAMPLES = 3  # minimum same-record, same-class non-missing beats needed
                       # before we trust a PATIENT-SPECIFIC median; below this,
                       # we fall back to the (training-only) global median instead

# Same fixed De Chazal et al. (2004) split used throughout this project.
TRAIN_RECORDS = [
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122", "124",
    "201", "203", "205", "207", "208", "209", "215", "220", "223", "230",
]
TEST_RECORDS = [
    "100", "103", "105", "111", "113", "117", "121", "123",
    "200", "202", "210", "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234",
]
EXCLUDED_RECORDS = ["102", "104", "107", "217"]  # paced-beat records, per AAMI convention


def split_by_patient(df: pd.DataFrame):
    """
    Same patient-level split as the strict version, but keeps the
    'record' and 'label' columns attached (needed by the imputation
    step below) rather than immediately trimming down to just the 9
    feature columns.
    """
    df = df.copy()
    df["record"] = df["record"].astype(str)

    train_df = df[df["record"].isin(TRAIN_RECORDS)].reset_index(drop=True)
    test_df = df[df["record"].isin(TEST_RECORDS)].reset_index(drop=True)

    excluded_beats = len(df) - len(train_df) - len(test_df)
    print(f"Excluded {excluded_beats} beats from paced-beat records {EXCLUDED_RECORDS} "
          f"(per AAMI convention)\n")

    return train_df, test_df


def compute_training_global_medians(train_df: pd.DataFrame) -> dict:
    """
    Computes the fallback reference value for each imputable feature,
    per AAMI class, using ONLY the training set's own real (non-NaN)
    values. This dict is the one and only place test-set information
    could leak in if we got this wrong - so it is built from train_df
    alone, and the exact same dict gets reused for both train and test
    imputation below.
    """
    return {
        feature: train_df.groupby("label")[feature].median()
        for feature in IMPUTABLE_FEATURES
    }


def impute_with_reference(df: pd.DataFrame, global_medians: dict, set_name: str) -> pd.DataFrame:
    """
    Fills missing qrs_duration / pr_interval / qt_interval values.

    Preferred reference: THIS PATIENT'S OWN median for that feature,
    among their other beats of the SAME AAMI class, in the SAME
    recording - safe to compute from either train or test on its own,
    since it never looks outside that one patient's own data.

    Fallback reference (used when a patient doesn't have at least
    MIN_LOCAL_SAMPLES of their own same-class beats): the global_medians
    passed in - which were computed from TRAINING patients only,
    regardless of whether this function is currently imputing the
    training set or the test set. This is what keeps test-set
    information from ever influencing anything.

    Adds an "n_imputed" column (0-3) so exactly how much of each set
    relies on imputed rather than measured values stays fully
    auditable.
    """
    df = df.copy()
    df["n_imputed"] = 0

    for feature in IMPUTABLE_FEATURES:
        missing = df[feature].isna()
        if not missing.any():
            continue

        group = df.groupby(["record", "label"])[feature]
        local_median = group.transform("median")
        local_count = group.transform(lambda s: s.notna().sum())

        # the SAME training-derived reference is used regardless of
        # which set (train or test) is currently being imputed
        fallback_series = df["label"].map(global_medians[feature])

        use_local = local_count >= MIN_LOCAL_SAMPLES
        fill_values = np.where(use_local, local_median, fallback_series)
        fill_series = pd.Series(fill_values, index=df.index)

        df.loc[missing, "n_imputed"] += 1
        df.loc[missing, feature] = fill_series[missing]

    # last-resort safety net, in case a class had no reference at all
    for feature in IMPUTABLE_FEATURES:
        if df[feature].isna().any():
            unresolved = df[feature].isna().sum()
            print(f"  WARNING: {unresolved} beats in {set_name} still missing {feature} "
                  f"after imputation - filling with this set's own overall median as a last resort")
            df[feature] = df[feature].fillna(df[feature].median())

    return df


def print_imputation_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Reports how much of each set ended up relying on imputed values, by class."""
    for name, split_df in [("TRAIN", train_df), ("TEST", test_df)]:
        n_imputed_beats = (split_df["n_imputed"] > 0).sum()
        print(f"\n{name}: {n_imputed_beats} of {len(split_df)} beats "
              f"({100 * n_imputed_beats / len(split_df):.1f}%) have at least one imputed feature")
        print(split_df.loc[split_df["n_imputed"] > 0, "label"].value_counts().to_string())


def drop_fully_synthetic_beats(df: pd.DataFrame, set_name: str) -> pd.DataFrame:
    """
    Drops any beat where ALL THREE imputable features (qrs_duration,
    pr_interval, qt_interval) were imputed - i.e. n_imputed == 3.

    This is a deliberate, narrow tightening on top of the relaxed rule:
    a beat with n_imputed of 1 or 2 still carries at least one REAL
    morphological measurement, contributing genuine signal alongside
    the imputed value(s). A beat with n_imputed == 3 carries NONE - its
    entire morphological profile is synthesized from other beats,
    contributing no real evidence at all. Dropping only this narrow
    worst-case group keeps the core benefit of relaxing the drop rule
    (retaining beats with partial real information) while removing the
    beats that were really just noise wearing the shape of data.
    """
    fully_synthetic = df["n_imputed"] == 3
    n_dropped = fully_synthetic.sum()

    if n_dropped > 0:
        print(f"\n{set_name}: dropping {n_dropped} beats with ZERO real morphological "
              f"measurements (n_imputed == 3), by class:")
        print(df.loc[fully_synthetic, "label"].value_counts().to_string())

    return df.loc[~fully_synthetic].reset_index(drop=True)


def print_dropped_summary(n_train_before: int, n_test_before: int, dropped_by_class: pd.DataFrame) -> None:
    """
    One-glance total of every beat removed by the n_imputed == 3 rule
    (train + test combined), so the headline number doesn't have to be
    added up by hand from the two separate per-set printouts above.
    """
    train_dropped = int(dropped_by_class["TRAIN"].sum()) if "TRAIN" in dropped_by_class else 0
    test_dropped = int(dropped_by_class["TEST"].sum()) if "TEST" in dropped_by_class else 0
    total_dropped = train_dropped + test_dropped
    total_before = n_train_before + n_test_before

    print("\n=== TOTAL beats dropped in this step (n_imputed == 3: no real morphological measurement) ===")
    print(f"  TOTAL dropped: {total_dropped} of {total_before} beats "
          f"({100 * total_dropped / total_before:.2f}%)")
    print(f"    TRAIN: {train_dropped} of {n_train_before} ({100 * train_dropped / n_train_before:.2f}%)")
    print(f"    TEST:  {test_dropped} of {n_test_before} ({100 * test_dropped / n_test_before:.2f}%)")
    print(f"  Beats retained: {total_before - total_dropped}")
    if total_dropped > 0:
        dropped_by_class = dropped_by_class.copy()
        dropped_by_class["TOTAL"] = dropped_by_class.sum(axis=1)
        print("  By class:")
        print(dropped_by_class.to_string())


def print_split_summary(y_train, y_test) -> None:
    """Same class-distribution summary as the strict version's Step 5."""
    print(f"\nTraining set: {len(y_train)} beats from {len(TRAIN_RECORDS)} records")
    print(f"Testing set:  {len(y_test)} beats\n")

    summary = pd.DataFrame({
        "train_count": y_train.value_counts(),
        "test_count": y_test.value_counts(),
    }).fillna(0).astype(int)
    summary["train_pct"] = 100 * summary["train_count"] / summary["train_count"].sum()
    summary["test_pct"] = 100 * summary["test_count"] / summary["test_count"].sum()

    print(summary.loc[["N", "S", "V", "F", "Q"]])


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    df = pd.read_csv("ecg_features_relaxed_raw.csv")

    print(f"Train records ({len(TRAIN_RECORDS)}): {TRAIN_RECORDS}")
    print(f"Test records  ({len(TEST_RECORDS)}): {TEST_RECORDS}\n")

    # 1) SPLIT FIRST - while qrs_duration/pr_interval/qt_interval may
    #    still contain NaNs.
    train_df, test_df = split_by_patient(df)

    # 2) Compute the fallback reference from TRAINING data only.
    global_medians = compute_training_global_medians(train_df)
    print("Training-only fallback medians (used for BOTH train and test imputation):")
    for feature, medians_by_class in global_medians.items():
        print(f"  {feature}: {medians_by_class.round(4).to_dict()}")

    # 3) IMPUTE SECOND - train and test each use their own local
    #    per-patient reference where possible, and the SAME
    #    training-derived fallback where not.
    train_df = impute_with_reference(train_df, global_medians, "TRAIN")
    test_df = impute_with_reference(test_df, global_medians, "TEST")

    print_imputation_summary(train_df, test_df)

    # 4) Final quality floor: drop beats with ZERO real morphological
    #    measurement left (n_imputed == 3) - see drop_fully_synthetic_beats()
    #    docstring for why this specific, narrow cut is worth making.
    n_train_before, n_test_before = len(train_df), len(test_df)
    dropped_by_class = pd.concat([
        train_df.loc[train_df["n_imputed"] == 3, "label"].value_counts().rename("TRAIN"),
        test_df.loc[test_df["n_imputed"] == 3, "label"].value_counts().rename("TEST"),
    ], axis=1).fillna(0).astype(int)
    train_df = drop_fully_synthetic_beats(train_df, "TRAIN")
    test_df = drop_fully_synthetic_beats(test_df, "TEST")
    print_dropped_summary(n_train_before, n_test_before, dropped_by_class)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["label"]

    print_split_summary(y_train, y_test)

    # Save the split so Step 5 (SMOTE) and Step 6 (Random Forest) can
    # load it directly. "_relaxed" suffix keeps these separate from the
    # strict version's saved files, so both remain available to compare.
    X_train.to_csv("X_train_relaxed.csv", index=False)
    X_test.to_csv("X_test_relaxed.csv", index=False)
    y_train.to_csv("y_train_relaxed.csv", index=False)
    y_test.to_csv("y_test_relaxed.csv", index=False)
    print("\nSaved X_train_relaxed.csv, X_test_relaxed.csv, y_train_relaxed.csv, y_test_relaxed.csv")

    # Companion audit files - same beats/order as above, but keeping
    # 'record' and 'n_imputed' so exactly how much of any given result
    # rests on imputed vs. measured values stays checkable later.
    train_df[["record", "label", "n_imputed"] + FEATURE_COLUMNS].to_csv("train_audit_relaxed.csv", index=False)
    test_df[["record", "label", "n_imputed"] + FEATURE_COLUMNS].to_csv("test_audit_relaxed.csv", index=False)
    print("Saved train_audit_relaxed.csv, test_audit_relaxed.csv (includes record + n_imputed for transparency)")
