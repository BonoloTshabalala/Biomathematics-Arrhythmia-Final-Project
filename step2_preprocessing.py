"""
STEP 2 — Digital Signal Pre-processing
=======================================
Goal: clean the raw ECG signal before we do anything else with it, using
the two-stage approach from the methodology:
  (a) remove baseline wander with a dual median filter
  (b) remove high-frequency noise with a Butterworth bandpass filter

Install what this script needs (in addition to Step 1's packages):
    pip install scipy
"""

import os
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt

DATA_DIR = "mitdb"


def _ms_to_odd_samples(window_ms: float, fs: int) -> int:
    """
    Converts a window length in milliseconds into a sample count, and
    forces it to be odd.

    Why odd? scipy's median_filter() centers its window on each point — it
    needs an equal number of samples on the left and right, which is
    only possible with an odd-sized window. If the raw conversion from
    ms to samples lands on an even number, we bump it up by 1.
    """
    n = int(round(window_ms / 1000 * fs))
    if n % 2 == 0:
        n += 1
    return n


def remove_baseline_wander(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Removes slow baseline drift (caused by breathing, patient movement,
    electrode shift) using the dual median filter described in the
    methodology.

    How a median filter works: slide a window across the signal, and at
    each point, replace the value with the MEDIAN of all samples inside
    that window. A median is robust to sharp spikes (like a QRS
    complex), so a wide-enough median filter effectively ignores the
    heartbeats entirely and traces out only the slow-moving baseline
    trend underneath them.

    Two passes, per the methodology:
      1. A 200 ms median filter — window narrow enough to still be
         pulled around a little by P-waves and T-waves, so...
      2. ...a second, wider 600 ms median filter smooths those
         remaining P/T-wave bumps out, leaving a clean baseline curve.

    That final baseline curve is then SUBTRACTED from the original
    signal, which flattens the drift while leaving the actual heartbeat
    shapes untouched.
    """
    window_1 = _ms_to_odd_samples(200, fs)
    window_2 = _ms_to_odd_samples(600, fs)

    # mode="nearest" tells the filter how to handle the very edges of the
    # signal, where the window would otherwise run off the array - it
    # just repeats the edge value instead, which is a standard, harmless
    # choice for a signal this long.
    baseline = median_filter(signal, size=window_1, mode="nearest")
    baseline = median_filter(baseline, size=window_2, mode="nearest")

    return signal - baseline


def bandpass_filter(signal: np.ndarray, fs: int, low_hz: float = 0.5,
                     high_hz: float = 45, order: int = 4) -> np.ndarray:
    """
    Removes high-frequency noise (50/60 Hz powerline interference,
    muscle tremor artifacts) using a 4th-order Butterworth bandpass
    filter, keeping only frequencies between 0.5 Hz and 45 Hz — the
    range that real cardiac electrical activity falls within.

    Two details worth knowing for Q&A:
      - "Bandpass" = keeps a band of frequencies, cuts everything below
        low_hz and above high_hz.
      - We use filtfilt() (not lfilter()) to make it "zero-phase." A
        normal filter shifts the signal slightly in time as a side
        effect, which would offset our R-peak locations relative to
        the annotation file. filtfilt() runs the filter forward, then
        backward, which cancels out that time shift — so the filtered
        signal lines up with the original in time, and our R-peak
        annotations stay valid.
    """
    nyquist = fs / 2  # the highest frequency we can represent at this sampling rate
    low = low_hz / nyquist
    high = high_hz / nyquist

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def preprocess_signal(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Runs the full Step 2 pipeline on one channel of raw ECG signal:
    baseline wander removal, followed by bandpass filtering.
    """
    no_wander = remove_baseline_wander(signal, fs)
    clean = bandpass_filter(no_wander, fs)
    return clean


def plot_before_after(record_name: str = "100", seconds: float = 5.0,
                       channel: int = 0, data_dir: str = DATA_DIR) -> None:
    """
    Loads one record, filters one channel of it, and plots the raw vs.
    cleaned signal side by side so we can visually confirm the filtering
    is doing what it should — flattening drift without distorting the
    QRS complexes.
    """
    record = wfdb.rdrecord(os.path.join(data_dir, record_name))
    fs = record.fs
    n_samples = int(seconds * fs)

    raw = record.p_signal[:n_samples, channel]
    clean = preprocess_signal(record.p_signal[:, channel], fs)[:n_samples]

    time_axis = [i / fs for i in range(n_samples)]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, sharey=True)
    axes[0].plot(time_axis, raw, color="gray", linewidth=0.8)
    axes[0].set_title(f"Raw signal — {record.sig_name[channel]}")
    axes[0].set_ylabel("mV")

    axes[1].plot(time_axis, clean, color="black", linewidth=0.8)
    axes[1].set_title("After baseline wander removal + bandpass filtering")
    axes[1].set_ylabel("mV")
    axes[1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(f"record_{record_name}_before_after.png", dpi=150)
    plt.show()  # pops up a window this time, since you're comparing two plots
    print(f"Saved plot to record_{record_name}_before_after.png")


if __name__ == "__main__":
    plot_before_after("100")
