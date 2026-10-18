"""
STEP 1 — Data Provenance
========================
Goal: download the MIT-BIH Arrhythmia Database from PhysioNet, load one
record to confirm everything works, and count the beats to check our
real class distribution (this settles the 89% vs 75% question with our
own numbers instead of quoting conflicting papers).

Before running, install the two packages this script needs:
    pip install wfdb matplotlib

wfdb = the official PhysioNet Python library. ECG files aren't plain
text, they're a binary format (.dat) plus header (.hea) and annotation
(.atr) files. wfdb knows how to parse all three, so we don't have to
write our own binary parser.
"""

import os
import wfdb
import matplotlib.pyplot as plt
from collections import Counter

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DATA_DIR = "mitdb"          # folder where the 48 records will be saved locally
DB_NAME = "mitdb"           # PhysioNet's short name for the MIT-BIH Arrhythmia Database


def download_mitbih_database(data_dir: str = DATA_DIR) -> None:
    """
    Downloads the full MIT-BIH Arrhythmia Database (48 records) from
    PhysioNet into a local folder.

    wfdb.dl_database() is a convenience function that:
      1. Looks up every record belonging to 'mitdb' on PhysioNet's servers.
      2. Downloads the .dat (raw signal), .hea (header/metadata), and
         .atr (cardiologist annotations) files for each one.
      3. Skips any record that's already downloaded, so it's safe to
         re-run this function without re-downloading everything.

    We only need to run this once — after that, everything lives on disk.
    """
    if os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0:
        print(f"'{data_dir}' already has files in it — skipping download.")
        return

    print("Downloading MIT-BIH Arrhythmia Database from PhysioNet...")
    wfdb.dl_database(DB_NAME, dl_dir=data_dir)
    print("Download complete.")


def load_record(record_name: str, data_dir: str = DATA_DIR):
    """
    Loads ONE record's signal + annotations from disk.

    A "record" in MIT-BIH = one 30-minute, 2-channel Holter recording
    from one subject (e.g. record '100', '101', ... '234').

    Returns two objects:
      record      -> wfdb.Record object. Holds the raw signal itself.
                      record.p_signal is a NumPy array of shape
                      (num_samples, num_channels) — the actual voltages.
                      record.fs is the sampling frequency (360 Hz here).
                      record.sig_name lists the channel names, e.g.
                      ['MLII', 'V1'].
      annotation  -> wfdb.Annotation object. Holds the cardiologists'
                      expert labels. annotation.sample is the array of
                      R-peak sample locations (WHERE each heartbeat is).
                      annotation.symbol is the array of beat-type codes
                      (WHAT each heartbeat is, e.g. 'N', 'V', 'A' — these
                      get mapped to the 5 AAMI classes in Step 3).

    This is exactly why we don't need our own R-peak detection algorithm
    (like Pan-Tompkins, which the lit review flagged as noise-sensitive):
    the annotation file already gives us cardiologist-verified R-peak
    locations for every beat in the recording.
    """
    record_path = os.path.join(data_dir, record_name)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, extension="atr")
    return record, annotation


def summarize_dataset(data_dir: str = DATA_DIR) -> None:
    """
    Loops through every downloaded record, tallies how many beats of
    each raw annotation symbol exist across the whole database, and
    prints a summary.

    This gives us the REAL class distribution from our actual data pull,
    which is what we should cite in the report/presentation instead of
    the conflicting 89% / 75% figures currently in different sections.
    (Note: this counts raw WFDB symbols, e.g. 'N', 'L', 'R', 'V', 'A' —
    Step 3 will map these onto the 5 AAMI superclasses N/S/V/F/Q.)
    """
    record_names = wfdb.get_record_list(DB_NAME)
    print(f"Found {len(record_names)} records in the database.")

    symbol_counts = Counter()
    total_beats = 0

    for name in record_names:
        _, annotation = load_record(name, data_dir)
        symbol_counts.update(annotation.symbol)
        total_beats += len(annotation.symbol)

    print(f"\nTotal annotated beats across all 48 records: {total_beats}")
    print("\nBeat symbol counts (most common first):")
    for symbol, count in symbol_counts.most_common():
        pct = 100 * count / total_beats
        print(f"  {symbol:>3}  {count:>6}  ({pct:5.2f}%)")


def plot_sample_beat_window(record_name: str = "100", seconds: float = 5.0,
                             data_dir: str = DATA_DIR) -> None:
    """
    Loads one record and plots the first few seconds of both ECG
    channels, with the cardiologist-annotated R-peaks marked on top.
    Useful as a sanity check that the download/loading worked, and as
    a figure for the presentation.
    """
    record, annotation = load_record(record_name, data_dir)
    fs = record.fs                         # sampling frequency, e.g. 360 Hz
    n_samples = int(seconds * fs)          # how many samples = 'seconds' worth of signal

    fig, axes = plt.subplots(record.p_signal.shape[1], 1, figsize=(10, 5), sharex=True)
    time_axis = [i / fs for i in range(n_samples)]

    for ch in range(record.p_signal.shape[1]):
        axes[ch].plot(time_axis, record.p_signal[:n_samples, ch], color="black", linewidth=0.8)

        # keep only the R-peak annotations that fall inside this time window
        peaks_in_window = [s for s in annotation.sample if s < n_samples]
        axes[ch].scatter([p / fs for p in peaks_in_window],
                          record.p_signal[peaks_in_window, ch],
                          color="red", zorder=3, label="Annotated R-peak")

        axes[ch].set_ylabel(f"{record.sig_name[ch]} (mV)")
        axes[ch].legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Record {record_name} — first {seconds}s, {len(record.p_signal)} total samples")
    plt.tight_layout()
    plt.savefig(f"record_{record_name}_sample.png", dpi=150)
    print(f"Saved plot to record_{record_name}_sample.png")


if __name__ == "__main__":
    # 1. Download all 48 records (only runs if not already downloaded)
    download_mitbih_database()

    # 2. Load one record as a smoke test, and print its basic properties
    record, annotation = load_record("100")
    print("\n--- Record 100 metadata ---")
    print(f"Sampling frequency: {record.fs} Hz")
    print(f"Channels: {record.sig_name}")
    print(f"Signal length: {record.p_signal.shape[0]} samples "
          f"({record.p_signal.shape[0] / record.fs / 60:.1f} minutes)")
    print(f"Number of annotated beats in this record: {len(annotation.sample)}")

    # 3. Count beat types across the whole database — settles the 89%/75% question
    summarize_dataset()

    # 4. Plot a few seconds of signal with R-peaks marked, as a visual sanity check
    plot_sample_beat_window("100")
