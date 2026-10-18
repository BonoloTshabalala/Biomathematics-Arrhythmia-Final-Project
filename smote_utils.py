"""
smote_utils.py — shared SMOTE strategy, importable from any script
======================================================================
Holds build_sampling_strategy() and its two constants in ONE shared
location, rather than inside step6_random_forest_gridsearch.py itself.

WHY THIS FILE NEEDS TO EXIST: when a trained model containing a SMOTE
step is saved with joblib, it doesn't save the function's actual code -
it saves a REFERENCE (which module, which name) and looks the function
up again when the model is loaded elsewhere. If build_sampling_strategy
lived inside step6's own file, that reference would point to "whichever
script is currently running" (Python calls this __main__) - which
breaks the moment a DIFFERENT script (like step7) tries to load the
saved model, since that function doesn't exist inside step7's file.

Putting it here instead gives it a stable, real module location
("smote_utils") that any script can import from and correctly find,
regardless of which script actually trains vs. loads the model.

This file must stay in the SAME FOLDER as every step6/7/8 script.
"""

from collections import Counter

CAPPED_CLASSES = ("F", "Q")
CAP_MULTIPLIER = 20


def build_sampling_strategy(y) -> dict:
    """
    Targeted/capped SMOTE strategy: full balance for S/V (enough real
    examples to support genuine diversity), capped 20x amplification
    for the data-scarce F/Q classes - see step5_smote_relaxed.py's
    original docstring for the full reasoning.

    Written as a function of y (not a fixed dict) so it can be handed
    to SMOTE's sampling_strategy parameter as a CALLABLE - imblearn
    calls this fresh every time SMOTE fits, which is what makes it
    safe to reuse inside cross-validation folds.
    """
    counts = Counter(y)
    majority_class = max(counts, key=counts.get)
    majority_count = counts[majority_class]

    strategy = {}
    for cls, count in counts.items():
        if cls == majority_class:
            continue
        if cls in CAPPED_CLASSES:
            strategy[cls] = min(majority_count, count * CAP_MULTIPLIER)
        else:
            strategy[cls] = majority_count
    return strategy
