"""
STEP 3 — Heartbeat Segmentation + AAMI Mapping
================================================
Goal: cut each 30-minute, 2-channel recording into individual heartbeats,
and map each beat's raw annotation symbol onto one of the 5 AAMI
superclasses (N, S, V, F, Q).

Each heartbeat is now BOTH channels, not just MLII — so one segmented
beat has shape (252 samples, 2 channels) instead of just (252,).

This file imports the cleaning functions we already built in Step 2,
so make sure step2_preprocessing.py is in the same folder as this file.
"""

import os
import time
import numpy as np
import wfdb
from collections import Counter

from step2_preprocessing import preprocess_signal  # reuse Step 2's filtering

DATA_DIR = "mitdb"

# Window around each R-peak: 90 samples before, 161 samples after (=> 252 total)
PRE_SAMPLES = 90
POST_SAMPLES = 161

# -----------------------------------------------------------------------
# AAMI mapping: raw WFDB annotation symbols -> 5 AAMI superclasses
# Any symbol NOT in this dictionary (e.g. '+', '~', '"', '[', ']', '|')
# is a non-beat marker (rhythm change, signal quality note, etc.) and
# gets dropped during segmentation rather than mapped.
# -----------------------------------------------------------------------
AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",   # Normal
    "A": "S", "a": "S", "J": "S", "S": "S",              # Supraventricular ectopic
    "V": "V", "E": "V",                                  # Ventricular ectopic
    "F": "F",                                            # Fusion
    "/": "Q", "f": "Q", "Q": "Q",                        # Unknown/paced
}


def preprocess_all_channels(p_signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Runs Step 2's filtering (baseline wander removal + bandpass) on
    EVERY channel of a record, not just one.

    p_signal has shape (num_samples, num_channels) — e.g. (650000, 2)
    for a record with MLII and V5. We loop over each column (channel),
    filter it independently, and rebuild the same 2D shape.

    Each channel gets filtered separately because they're two different
    physical electrodes recording two different views of the heart —
    there's no meaningful way to combine them before filtering, only
    after (which is what segmentation below does).
    """
    n_channels = p_signal.shape[1]
    filtered = np.zeros_like(p_signal)

    for ch in range(n_channels):
        filtered[:, ch] = preprocess_signal(p_signal[:, ch], fs)

    return filtered


def segment_record(record_name: str, data_dir: str = DATA_DIR):
    """
    Loads one record, filters both channels, then cuts out one 252-sample
    window per valid heartbeat, keeping both channels for each beat.

    Returns:
      beats   -> array of shape (num_beats, 252, num_channels)
                 e.g. (2200, 252, 2) for a typical record
      labels  -> array of shape (num_beats,) with AAMI class strings,
                 e.g. ['N', 'N', 'S', 'N', ...]
      r_peaks -> array of shape (num_beats,) with the R-peak's sample
                 position in the ORIGINAL full-length recording (not
                 its position inside the 252-sample window). Step 4
                 needs these exact positions to measure RR-intervals
                 and to locate the P/QRS/T waves around each beat.

    Two kinds of beats get skipped:
      1. Symbols not in AAMI_MAP (non-beat markers, see comment above).
      2. "Edge beats" — the very first/last beat in a recording, if its
         252-sample window would run off the start or end of the signal
         (e.g. a beat 30 samples from the start can't have 90 samples
         of "before" context). These are rare (at most 1-2 per record)
         and simply can't be windowed, so we drop them.
    """
    record = wfdb.rdrecord(os.path.join(data_dir, record_name))
    annotation = wfdb.rdann(os.path.join(data_dir, record_name), extension="atr")

    fs = record.fs
    n_samples, n_channels = record.p_signal.shape

    clean_signal = preprocess_all_channels(record.p_signal, fs)

    beats = []
    labels = []
    r_peaks = []

    for r_peak, symbol in zip(annotation.sample, annotation.symbol):
        if symbol not in AAMI_MAP:
            continue  # not a real beat label, skip

        start = r_peak - PRE_SAMPLES
        end = r_peak + POST_SAMPLES + 1  # +1 because slicing is exclusive at the end

        if start < 0 or end > n_samples:
            continue  # edge beat, window would run off the recording

        window = clean_signal[start:end, :]  # shape: (252, n_channels)
        beats.append(window)
        labels.append(AAMI_MAP[symbol])
        r_peaks.append(r_peak)

    return np.array(beats), np.array(labels), np.array(r_peaks)


def build_dataset(record_names=None, data_dir: str = DATA_DIR):
    """
    Runs segment_record() across many records and stacks the results
    into one big dataset.

    Returns:
      X -> shape (total_beats, 252, num_channels)
      y -> shape (total_beats,) AAMI class labels
      groups -> shape (total_beats,) which record each beat came from
                (this is what makes the inter-patient train/test split
                possible later — we need to know which patient each
                beat belongs to)
      r_peaks -> shape (total_beats,) R-peak sample position in each
                 beat's original recording (needed for Step 4)
    """
    if record_names is None:
        record_names = wfdb.get_record_list("mitdb")

    all_beats, all_labels, all_groups, all_r_peaks = [], [], [], []

    for i, name in enumerate(record_names, start=1):
        start_time = time.time()
        beats, labels, r_peaks = segment_record(name, data_dir)
        elapsed = time.time() - start_time

        all_beats.append(beats)
        all_labels.append(labels)
        all_groups.extend([name] * len(labels))
        all_r_peaks.append(r_peaks)
        print(f"[{i}/{len(record_names)}] Record {name}: "
              f"{len(labels)} usable beats ({elapsed:.1f}s)")

    X = np.concatenate(all_beats, axis=0)
    y = np.concatenate(all_labels, axis=0)
    groups = np.array(all_groups)
    r_peaks = np.concatenate(all_r_peaks, axis=0)

    return X, y, groups, r_peaks


if __name__ == "__main__":
    X, y, groups, r_peaks = build_dataset()

    print(f"\nFinal dataset shape: {X.shape}")   # (total_beats, 252, 2)
    print(f"Labels shape: {y.shape}")

    print("\nTrue AAMI class distribution (after dropping non-beat markers and edge beats):")
    counts = Counter(y)
    total = len(y)
    for cls in ["N", "S", "V", "F", "Q"]:
        count = counts.get(cls, 0)
        print(f"  {cls}: {count:>6}  ({100 * count / total:5.2f}%)")

    # Save to disk so Step 4 (feature extraction) can load this without
    # re-running segmentation from scratch every time.
    np.save("X_beats.npy", X)
    np.save("y_labels.npy", y)
    np.save("groups.npy", groups)
    np.save("r_peaks.npy", r_peaks)
    print("\nSaved X_beats.npy, y_labels.npy, groups.npy, r_peaks.npy")
