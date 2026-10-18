"""
STEP 4 (RELAXED) — Feature Extraction with Partial-Feature Retention
========================================================================
Same as the strict version, with one deliberate change to how a beat
gets dropped, split across TWO files:

  THIS FILE (Step 4): drops a beat only if a REQUIRED feature is
    missing (pre_rr, post_rr, local_avg_rr, skewness, kurtosis,
    p_amplitude - all under 1% missing). qrs_duration, pr_interval, and
    qt_interval are left as NaN where missing - NOT dropped, NOT
    imputed here.

  step5_train_test_split_relaxed.py: splits the data into train/test
    FIRST, then imputes the remaining qrs_duration/pr_interval/
    qt_interval gaps SECOND - using each patient's own same-class
    median where possible, falling back to a median computed from
    TRAINING patients only (never test patients) where not. See that
    file for the full reasoning.

WHY IMPUTATION IS NOT DONE HERE: if we imputed before splitting, the
fallback reference (used when a patient doesn't have enough of their
own same-class beats) would be computed by pooling ALL 48 patients
together - including future test patients. That's a data leak: test-set
information would quietly influence values used to train the model.
Splitting first and imputing second (with the fallback computed from
training patients only) avoids this entirely - the same principle as
fitting a scikit-learn scaler/imputer only on training data, then
applying it to the test set.

THE PROBLEM WITH "ALL-OR-NOTHING" (why we're doing this at all):
dropping a beat because ANY ONE of its 9 features is missing
disproportionately removes V/F/Q beats (the classes with distorted
morphology that's hardest to delineate). Because patient labels
(Step 3b) were computed from the FULL, undropped beat set, but the
beat-level classifier only ever sees the SURVIVORS of this drop, there
was a real mismatch: the model would train on - and later predict
from - a biased slice of each patient's true arrhythmic evidence.
"""

import os
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.stats import skew, kurtosis

from step2_preprocessing import preprocess_signal

# Stop pandas from truncating columns in printed previews
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

DATA_DIR = "mitdb"
FS = 360  # MIT-BIH's fixed sampling frequency

MLII_CHANNEL = 0  # which channel morphological features are measured from


def compute_rr_features(r_peaks: np.ndarray) -> pd.DataFrame:
    """
    Computes the 3 time-domain features for every beat in one record,
    purely from R-peak sample positions (no signal values needed).

      Pre-RR:  time since the PREVIOUS beat's R-peak (seconds)
      Post-RR: time until the NEXT beat's R-peak (seconds)
      Local average RR: mean of the 10 most recent Pre-RR intervals —
        a rolling sense of "this patient's normal heart rate right now,"
        which helps the model spot beats that break the local rhythm.

    r_peaks must be in time order (which they are, coming from Step 3).
    The very first beat in a record has no "previous" beat, and the
    very last has no "next" beat — those get filled with NaN and
    dropped later, since a feature can't be computed without them.
    """
    rr_intervals = np.diff(r_peaks) / FS  # seconds between consecutive R-peaks

    pre_rr = np.full(len(r_peaks), np.nan)
    post_rr = np.full(len(r_peaks), np.nan)
    local_avg_rr = np.full(len(r_peaks), np.nan)

    pre_rr[1:] = rr_intervals          # beat i's pre-RR = interval (i-1 -> i)
    post_rr[:-1] = rr_intervals        # beat i's post-RR = interval (i -> i+1)

    for i in range(len(r_peaks)):
        # average of up to the 10 most recent pre-RR intervals available so far
        recent = pre_rr[max(0, i - 9):i + 1]
        recent = recent[~np.isnan(recent)]
        if len(recent) > 0:
            local_avg_rr[i] = np.mean(recent)

    return pd.DataFrame({
        "pre_rr": pre_rr,
        "post_rr": post_rr,
        "local_avg_rr": local_avg_rr,
    })


def compute_morphological_features(clean_channel: np.ndarray, r_peaks: np.ndarray) -> pd.DataFrame:
    """
    Computes the 4 morphological features for every beat in one record,
    using neurokit2 to locate each beat's P-wave, QRS complex, and
    T-wave boundaries on the cleaned MLII channel.

    nk.ecg_delineate() returns the sample position of each fiducial
    point (P-onset, P-peak, QRS-onset, QRS-offset, T-offset, etc.) for
    every R-peak we give it. From those positions we compute:

      QRS duration:  (QRS-offset - QRS-onset) / fs
      PR interval:   (QRS-onset - P-onset) / fs
      QT interval:   (T-offset - QRS-onset) / fs
      P-wave amplitude: the signal's voltage AT the P-peak sample

    FALLBACK DELINEATION (recovery attempt for dropped beats):
    We first run "dwt" (wavelet-based) on every beat - this is generally
    the most accurate method when it succeeds. Then, instead of
    re-running a second method on the WHOLE record again, we only
    retry the SPECIFIC beats dwt failed on, using "cwt" (a different
    wavelet transform that computes the same onset/offset points dwt
    does, but via a different algorithm, so it often succeeds on a
    different set of beats than dwt). Only retrying the failures - not
    every beat again - is what keeps this fast: dwt already handles the
    large majority of beats, so cwt only ever has a small remainder to
    work through.

    Note: neurokit2's "peak" method was tried during development but
    dropped from this pipeline. It only locates wave PEAKS, not
    onset/offset boundaries, so it can't recover qrs_duration,
    pr_interval, or qt_interval at all (our three most-affected
    features) - it could only ever help p_amplitude, which barely has a
    missingness problem (0.65%) to begin with. Its internal per-beat Q
    detection is also noticeably slower than dwt/cwt. Given it cost a
    lot and could only help a feature that didn't need much help, it
    wasn't worth keeping in the fallback chain.

    A method's result is NEVER overwritten where an earlier method
    already succeeded - this is deliberately conservative: it can only
    recover beats, never change a measurement that already worked, and
    we track exactly which values were recovered this way so it's
    fully auditable rather than a silent substitution.
    """
    KEYS = ["ECG_P_Onsets", "ECG_P_Peaks", "ECG_R_Onsets", "ECG_R_Offsets", "ECG_T_Offsets"]
    n = len(r_peaks)

    def run_delineation(peak_indices):
        """Runs dwt-style delineation on just the given r_peaks, returns a dict of arrays."""
        if len(peak_indices) == 0:
            return {}
        try:
            _, waves = nk.ecg_delineate(
                clean_channel, rpeaks=r_peaks[peak_indices], sampling_rate=FS, method="cwt",
            )
            return waves
        except Exception:
            return {}

    # --- Pass 1: dwt on every beat ---
    try:
        _, waves_dwt = nk.ecg_delineate(clean_channel, rpeaks=r_peaks, sampling_rate=FS, method="dwt")
    except Exception:
        waves_dwt = {}

    result = {key: np.full(n, np.nan) for key in KEYS}
    for key in KEYS:
        values = np.array(waves_dwt.get(key, [np.nan] * n), dtype=float)
        if len(values) == n:
            result[key] = values

    recovered = np.zeros(n, dtype=bool)

    # --- Pass 2: cwt, but ONLY on beats still missing at least one needed point ---
    still_missing_mask = np.zeros(n, dtype=bool)
    for key in KEYS:
        still_missing_mask |= np.isnan(result[key])
    failed_indices = np.where(still_missing_mask)[0]

    waves_cwt = run_delineation(failed_indices)
    for key in KEYS:
        values = np.array(waves_cwt.get(key, []), dtype=float)
        if len(values) != len(failed_indices):
            continue  # length mismatch on this small batch - skip this key safely
        for local_i, global_i in enumerate(failed_indices):
            if np.isnan(result[key][global_i]) and not np.isnan(values[local_i]):
                result[key][global_i] = values[local_i]
                recovered[global_i] = True

    p_onsets, p_peaks = result["ECG_P_Onsets"], result["ECG_P_Peaks"]
    qrs_onsets, qrs_offsets, t_offsets = result["ECG_R_Onsets"], result["ECG_R_Offsets"], result["ECG_T_Offsets"]
    recovered_by_fallback = recovered

    qrs_duration = (qrs_offsets - qrs_onsets) / FS
    pr_interval = (qrs_onsets - p_onsets) / FS
    qt_interval = (t_offsets - qrs_onsets) / FS

    # P-wave amplitude = the actual voltage at the P-peak sample.
    # Indexing with NaN isn't possible, so we look up valid positions
    # only and leave the rest as NaN.
    p_amplitude = np.full(len(r_peaks), np.nan)
    valid = ~np.isnan(p_peaks)
    valid_indices = p_peaks[valid].astype(int)
    p_amplitude[valid] = clean_channel[valid_indices]

    return pd.DataFrame({
        "qrs_duration": qrs_duration,
        "pr_interval": pr_interval,
        "qt_interval": qt_interval,
        "p_amplitude": p_amplitude,
        "recovered_by_fallback": recovered_by_fallback,
    })


def compute_statistical_features(beats: np.ndarray) -> pd.DataFrame:
    """
    Computes the 2 statistical features directly from each beat's raw
    252-sample MLII window (no external library needed for this part).

      Skewness: how lopsided the beat's voltage distribution is
        (a symmetric waveform has skewness near 0).
      Kurtosis: how "peaked vs. flat" the distribution is (a sharp,
        spiky QRS complex has high kurtosis; a smooth, gentle beat
        has lower kurtosis).

    beats has shape (num_beats, 252, num_channels) — we take the MLII
    channel only, one row per beat.
    """
    mlii_beats = beats[:, :, MLII_CHANNEL]  # shape: (num_beats, 252)

    return pd.DataFrame({
        "skewness": skew(mlii_beats, axis=1),
        "kurtosis": kurtosis(mlii_beats, axis=1),
    })


def extract_features_for_record(beats: np.ndarray, r_peaks: np.ndarray,
                                  full_clean_channel: np.ndarray) -> pd.DataFrame:
    """
    Combines all 9 features for every beat belonging to ONE record into
    a single table (one row per beat, one column per feature).
    """
    rr_df = compute_rr_features(r_peaks)
    morph_df = compute_morphological_features(full_clean_channel, r_peaks)
    stat_df = compute_statistical_features(beats)

    return pd.concat([rr_df, morph_df, stat_df], axis=1)


# --- Parallel processing setup ---
# Each of the 48 records is fully independent of every other, so instead
# of processing them one at a time on a single CPU core (which is what
# was making this so slow), we hand each record to a separate worker
# process and let them run simultaneously across all your CPU cores.
#
# _WORKER_DATA loads the big shared arrays (X_beats.npy etc.) ONCE per
# worker process when it starts up, rather than once per record - with
# 48 records but typically far fewer CPU cores, this avoids re-loading
# a ~400MB file dozens of times over.
_WORKER_DATA = {}


def _init_worker():
    # mmap_mode="r" is the important part here: instead of every worker
    # process loading its own full private copy of X_beats.npy (this file
    # alone is roughly 440 MB), memory-mapping lets the operating system
    # share the same underlying file pages read-only across all worker
    # processes. Without this, N worker processes could mean N separate
    # ~440MB copies in RAM simultaneously - easily enough to exhaust
    # memory on a laptop and cause a silent crash with no Python
    # traceback at all (Windows just kills the process outright).
    _WORKER_DATA["X"] = np.load("X_beats.npy", mmap_mode="r")
    _WORKER_DATA["y"] = np.load("y_labels.npy", allow_pickle=True)
    _WORKER_DATA["groups"] = np.load("groups.npy", allow_pickle=True)
    _WORKER_DATA["r_peaks"] = np.load("r_peaks.npy")


def process_record(record_name):
    """
    Does everything Step 4 needs for ONE record: pulls out just that
    record's beats/labels/R-peaks, re-derives its cleaned signal, and
    extracts all 9 features. Runs inside a worker process, so it must
    be a plain top-level function (not nested inside another function)
    for Python's multiprocessing to be able to send it to a worker.
    """
    import time
    import wfdb

    start = time.time()
    X, y, groups, r_peaks_all = (_WORKER_DATA["X"], _WORKER_DATA["y"],
                                  _WORKER_DATA["groups"], _WORKER_DATA["r_peaks"])

    mask = groups == record_name
    record_beats = X[mask]
    record_r_peaks = r_peaks_all[mask]
    record_labels = y[mask]  # pulled per-record now, not assigned globally at the end -
                              # this guarantees correct label alignment regardless of
                              # what order records finish processing in

    raw_record = wfdb.rdrecord(os.path.join(DATA_DIR, record_name))
    full_clean_channel = preprocess_signal(raw_record.p_signal[:, MLII_CHANNEL], FS)

    record_features = extract_features_for_record(record_beats, record_r_peaks, full_clean_channel)
    record_features["record"] = record_name
    record_features["label"] = record_labels

    elapsed = time.time() - start
    return record_name, record_features, elapsed


REQUIRED_FEATURES = ["pre_rr", "post_rr", "local_avg_rr", "skewness", "kurtosis", "p_amplitude"]
# Note: qrs_duration / pr_interval / qt_interval are NOT dropped here even
# when missing - imputing them now happens in Step 5, AFTER the train/test
# split, so that any fallback reference value is computed from training
# patients only (see step5_train_test_split_relaxed.py for why this order
# matters and the full explanation).


if __name__ == "__main__":
    import time

    groups_all = np.load("groups.npy", allow_pickle=True)
    record_names = list(np.unique(groups_all))

    # QUICK TEST MODE: set this to a small number (e.g. 3) to run on just
    # a handful of records as a fast sanity check that everything works -
    # useful for your lecturer to quickly confirm the pipeline runs
    # correctly without waiting ~1 hour for the full 48-record run.
    # Set back to None for the full, real run.
    QUICK_TEST_LIMIT = None
    if QUICK_TEST_LIMIT is not None:
        record_names = record_names[:QUICK_TEST_LIMIT]
        print(f"QUICK TEST MODE: only processing the first {QUICK_TEST_LIMIT} records.\n")

    _init_worker()  # loads X_beats.npy etc. once, into this one process

    print(f"Processing {len(record_names)} records sequentially...")
    t0 = time.time()

    all_features = []
    failed_records = []
    for i, record_name in enumerate(record_names, start=1):
        try:
            record_name, record_features, elapsed = process_record(record_name)
            all_features.append(record_features)
            print(f"[{i}/{len(record_names)}] Record {record_name}: "
                  f"{len(record_features)} beats ({elapsed:.1f}s)")
        except Exception as e:
            # A single record failing (corrupt file, unexpected data quirk,
            # environment issue, etc.) should not take down the entire run,
            # especially when someone else is the one running this. We log
            # it clearly and keep going with the remaining records.
            print(f"[{i}/{len(record_names)}] Record {record_name}: FAILED - {type(e).__name__}: {e}")
            failed_records.append(record_name)

    print(f"\nAll records processed in {time.time() - t0:.1f}s total")
    if failed_records:
        print(f"WARNING: {len(failed_records)} record(s) failed and were skipped: {failed_records}")

    features_df = pd.concat(all_features, ignore_index=True)
    # (labels are already attached per-record inside process_record now,
    # so there's no separate features_df["label"] = y assignment here)

    # Save the RAW table (before dropping anything) — this keeps every
    # NaN in place, which is exactly what the diagnostic script needs
    # to figure out which feature and which AAMI class are causing beats
    # to be dropped.
    features_df.to_csv("ecg_features_raw.csv", index=False)

    # --- Report how much the dwt-to-cwt fallback recovered (unchanged from strict version) ---
    feature_cols = ["qrs_duration", "pr_interval", "qt_interval", "p_amplitude"]
    would_be_complete = features_df[feature_cols].notna().all(axis=1)
    recovered_and_complete = would_be_complete & features_df["recovered_by_fallback"]

    print(f"\nBeats where the fallback filled in at least one point: "
          f"{features_df['recovered_by_fallback'].sum()}")
    print(f"Of those, beats that are now FULLY complete thanks to the fallback: "
          f"{recovered_and_complete.sum()}")
    features_df = features_df.drop(columns=["recovered_by_fallback"])

    # --- RELAXED rule: only drop a beat if a REQUIRED feature is missing ---
    # (under 1% for every feature in this list). qrs_duration / pr_interval /
    # qt_interval are deliberately left with NaNs still in them here - Step 5
    # is where those get imputed, AFTER the train/test split, so the
    # fallback reference values never see test-patient data.
    before = len(features_df)
    features_df = features_df.dropna(subset=REQUIRED_FEATURES).reset_index(drop=True)
    print(f"\nDropped {before - len(features_df)} beats missing a REQUIRED feature "
          f"({len(features_df)} beats remain)")

    still_missing = features_df[["qrs_duration", "pr_interval", "qt_interval"]].isna().any(axis=1).sum()
    print(f"{still_missing} of these beats still have at least one missing "
          f"qrs_duration/pr_interval/qt_interval value - these are NOT dropped "
          f"here, they'll be imputed in Step 5 after the train/test split.")

    print("\nFeature table preview:")
    print(features_df.head())

    features_df.to_csv("ecg_features_relaxed_raw.csv", index=False)
    print("\nSaved ecg_features_relaxed_raw.csv "
          "(still has some NaNs in qrs_duration/pr_interval/qt_interval - "
          "run step5_train_test_split_relaxed.py next, which splits first "
          "and imputes second)")
