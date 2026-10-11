"""
Plot the confusion matrix for the saved model.

Put this next to train.py (your classification script) and best_model.joblib, then run:
    python plot_confusion.py path/to/iBeacon_RSSI_Labeled.csv
"""
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Reuses the loading, cleaning and split logic from the training script.
# If your script has a different file name, change "train" below.
from train import load, block_split, to_xy, CSV_PATH, MISSING_FILL

MODEL_PATH = "best_model.joblib"
OUT_PATH = "confusion_matrix.png"


def main():
    df, loc_col, date_col, beacon_cols = load(CSV_PATH)
    X = df[beacon_cols].replace(-200, MISSING_FILL)
    y = df[loc_col]

    # The split is deterministic, so this recreates the same test rows
    _, test_mask, _ = block_split(df, loc_col)
    X_test, y_test = X[test_mask], y[test_mask]

    model = joblib.load(MODEL_PATH)
    pred = model.predict(X_test)

    # Order labels by grid position so physical neighbours sit close together
    labels = sorted(set(model.classes_) | set(y_test), key=to_xy)
    cm = confusion_matrix(y_test, pred, labels=labels)

    # Draw zeros as white so only real counts light up
    shown = np.ma.masked_equal(cm, 0)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("white")

    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(shown, cmap=cmap, interpolation="nearest")
    step = 5
    ticks = list(range(0, len(labels), step))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([labels[i] for i in ticks], rotation=90, fontsize=7)
    ax.set_yticklabels([labels[i] for i in ticks], fontsize=7)
    ax.set_xlabel("Predicted location")
    ax.set_ylabel("True location")
    ax.set_title(f"Confusion matrix ({len(y_test)} test rows, {len(labels)} locations)")
    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved {OUT_PATH}")

    # Most common mix-ups, as text
    pairs = [
        (cm[i, j], labels[i], labels[j])
        for i in range(len(labels))
        for j in range(len(labels))
        if i != j and cm[i, j] > 0
    ]
    pairs.sort(reverse=True)
    print("\nTop 10 mix-ups (count, true -> predicted):")
    for count, t, p in pairs[:10]:
        print(f"  {count:2d}  {t} -> {p}")


if __name__ == "__main__":
    main()