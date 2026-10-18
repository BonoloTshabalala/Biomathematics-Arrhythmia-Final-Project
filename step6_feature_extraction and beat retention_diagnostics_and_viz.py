"""
STEP 4b — Diagnostics + Feature Visualization
===============================================
Goal: figure out exactly which feature is causing beats to be dropped
(and whether it's concentrated in specific AAMI classes), and produce
visual plots of all 9 features for the presentation.

Uses ecg_features_raw.csv — the UN-dropped version Step 4 now saves,
which still has every NaN in place.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Stop pandas from truncating rows/columns when we print tables —
# so nothing gets hidden with "..." in the terminal output.
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", None)

FEATURE_COLUMNS = [
    "pre_rr", "post_rr", "local_avg_rr",
    "qrs_duration", "pr_interval", "qt_interval", "p_amplitude",
    "skewness", "kurtosis",
]


def diagnose_missing_values(df: pd.DataFrame) -> None:
    """
    Prints two things:
      1. What % of ALL beats are missing each individual feature —
         tells us WHICH feature is the main culprit.
      2. What % of beats are missing at least one feature, broken down
         BY AAMI class — tells us whether the drops are concentrated in
         certain arrhythmia types (a real, explainable finding) or
         spread evenly (more likely a general signal-quality issue).
    """
    total = len(df)

    print("=== Missing values per feature (out of all beats) ===")
    missing_per_feature = df[FEATURE_COLUMNS].isna().sum().sort_values(ascending=False)
    for feature, count in missing_per_feature.items():
        print(f"  {feature:<15} {count:>6} missing  ({100 * count / total:5.2f}%)")

    print("\n=== % of beats with at least one missing feature, by AAMI class ===")
    df = df.copy()
    df["any_missing"] = df[FEATURE_COLUMNS].isna().any(axis=1)
    by_class = df.groupby("label")["any_missing"].agg(["sum", "count"])
    by_class["pct_missing"] = 100 * by_class["sum"] / by_class["count"]
    by_class.columns = ["beats_dropped", "total_beats", "pct_dropped"]
    print(by_class.sort_values("pct_dropped", ascending=False))


def diagnose_fallback_recovery(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reports how many beats are dropped vs. retained under:
      1. DWT only
      2. DWT + CWT fallback (what the pipeline actually does)
      3. The difference = beats recovered by the CWT fallback

    "Dropped" here means the STRICT rule: a beat is dropped if ANY of the
    9 features is missing (the all-or-nothing rule the relaxed pipeline
    was built to improve on).

    How DWT-only is reconstructed from ecg_features_raw.csv: the CWT
    fallback only ever FILLS points that DWT left empty, and
    'recovered_by_fallback' flags every beat where it filled at least one.
    All 5 delineation points feed at least one of the 4 morphological
    features, so:
      - complete NOW and recovered_by_fallback  -> was incomplete under DWT only
      - complete NOW and not recovered          -> complete under DWT only too
      - incomplete NOW                          -> incomplete under DWT only too
    So the beat-level totals below are exact, not estimates.
    """
    if "recovered_by_fallback" not in df.columns:
        print("\n[skipped] 'recovered_by_fallback' column not found in the CSV - "
              "re-run feature extraction so ecg_features_raw.csv includes it.")
        return pd.DataFrame()

    total = len(df)
    complete_final = df[FEATURE_COLUMNS].notna().all(axis=1)
    recovered = df["recovered_by_fallback"].astype(bool)

    complete_dwt = complete_final & ~recovered   # retained under DWT only
    gained = complete_final & recovered          # retained ONLY thanks to CWT

    retained_dwt, retained_both = int(complete_dwt.sum()), int(complete_final.sum())
    dropped_dwt, dropped_both = total - retained_dwt, total - retained_both

    print("\n=== Beats dropped vs. retained: DWT only vs. DWT + CWT fallback ===")
    print(f"  {'':<22}{'Retained':>10}{'Dropped':>10}{'% retained':>12}")
    print(f"  {'DWT only':<22}{retained_dwt:>10}{dropped_dwt:>10}{100 * retained_dwt / total:>11.2f}%")
    print(f"  {'DWT + CWT fallback':<22}{retained_both:>10}{dropped_both:>10}{100 * retained_both / total:>11.2f}%")
    print(f"  {'Difference (recovered)':<22}{retained_both - retained_dwt:>+10}{dropped_both - dropped_dwt:>+10}"
          f"{100 * (retained_both - retained_dwt) / total:>+11.2f}%")

    print("\n=== Beats recovered by the CWT fallback, by AAMI class ===")
    by_class = pd.DataFrame({
        "total_beats": df.groupby("label").size(),
        "dropped_dwt_only": (~complete_dwt).groupby(df["label"]).sum(),
        "dropped_dwt_cwt": (~complete_final).groupby(df["label"]).sum(),
        "recovered": gained.groupby(df["label"]).sum(),
    })
    by_class["pct_of_dwt_drops_recovered"] = (
        100 * by_class["recovered"] / by_class["dropped_dwt_only"].replace(0, pd.NA)
    ).astype(float).round(2)
    print(by_class)
    return by_class


def plot_feature_distributions(df: pd.DataFrame) -> None:
    """
    Plots a histogram of every feature, one subplot per feature, split
    by AAMI class (so you can visually see whether e.g. Ventricular
    beats really do have wider QRS durations than Normal beats).

    Rows with any missing feature are dropped just for this plot, since
    a histogram can't include NaN.
    """
    clean_df = df.dropna(subset=FEATURE_COLUMNS)
    classes = sorted(clean_df["label"].unique())

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i, feature in enumerate(FEATURE_COLUMNS):
        for cls in classes:
            values = clean_df.loc[clean_df["label"] == cls, feature]
            axes[i].hist(values, bins=40, alpha=0.5, label=cls, density=True)
        axes[i].set_title(feature)
        axes[i].legend(fontsize=8)

    fig.suptitle("Feature distributions by AAMI class", fontsize=14)
    plt.tight_layout()
    plt.savefig("feature_distributions.png", dpi=150)
    plt.show()
    print("\nSaved feature_distributions.png")


def diagnose_missing_by_class_and_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crosses the two breakdowns from diagnose_missing_values() together:
    for EACH AAMI class, what % of ITS beats are missing EACH individual
    feature. This is the missing piece from the two separate tables
    above - it tells us not just "V drops a lot" and "qt_interval fails
    a lot" separately, but whether V's problem is specifically
    qt_interval, specifically qrs_duration, or spread across several -
    which is exactly what you'd want to know before targeting any
    further fix at a specific class/feature combination.

    Only the 4 morphological features are included here (the ones that
    actually have meaningful missingness) - pre_rr/post_rr/local_avg_rr/
    skewness/kurtosis are all under 0.1% missing for every class, so
    including them would just add noise to the table.
    """
    morph_features = ["qrs_duration", "pr_interval", "qt_interval", "p_amplitude"]

    print("\n=== % missing, BY CLASS x BY FEATURE (the crossed breakdown) ===")
    crossed = df.groupby("label")[morph_features].apply(lambda g: 100 * g.isna().mean())
    crossed = crossed.round(2)
    print(crossed)

    return crossed


def plot_missing_heatmap(crossed: pd.DataFrame) -> None:
    """
    Plots the class x feature missingness table above as a heatmap -
    darker cells = more of that class's beats are missing that specific
    feature. Makes it immediate to spot which class/feature combination
    is the biggest remaining problem, rather than scanning a table of
    numbers.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(crossed.values, cmap="Reds", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(crossed.columns)))
    ax.set_xticklabels(crossed.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(crossed.index)))
    ax.set_yticklabels(crossed.index)

    for i in range(crossed.shape[0]):
        for j in range(crossed.shape[1]):
            value = crossed.values[i, j]
            text_color = "white" if value > 50 else "black"
            ax.text(j, i, f"{value:.1f}%", ha="center", va="center", color=text_color, fontsize=10)

    ax.set_title("% of beats missing each feature, by AAMI class")
    fig.colorbar(im, ax=ax, label="% missing")
    plt.tight_layout()
    plt.savefig("missing_by_class_and_feature.png", dpi=150)
    plt.show()
    print("Saved missing_by_class_and_feature.png")


if __name__ == "__main__":
    df = pd.read_csv("ecg_features_raw.csv")

    print(f"Total beats (before dropping): {len(df)}\n")

    diagnose_missing_values(df)
    diagnose_fallback_recovery(df)
    crossed = diagnose_missing_by_class_and_feature(df)
    plot_missing_heatmap(crossed)
    plot_feature_distributions(df)
