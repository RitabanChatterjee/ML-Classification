"""
BLE RSSI indoor localization: classification with a per-location block split.

Usage:
    python ble_classification.py path/to/iBeacon_RSSI_Labeled.csv
"""
import joblib
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "iBeacon_RSSI_Labeled.csv"
TEST_FRAC = 0.25      # last 25% of each location's rows go to test
GAP = 1               # rows dropped between train and test at the boundary
MIN_ROWS = 4          # locations with fewer rows stay in train only
BLOCK_SIZE = 2        # consecutive rows per block (used to group CV folds)
MISSING_FILL = -100   # replaces the -200 "out of range" placeholder
SEED = 42


def load(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    loc_col = next(c for c in df.columns if c.lower() == "location")
    date_col = next(c for c in df.columns if c.lower() == "date")
    beacon_cols = [c for c in df.columns if c.lower().startswith("b30")]

    df[loc_col] = df[loc_col].astype(str).str.strip()
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
    df = df.sort_values(date_col).reset_index(drop=True)
    return df, loc_col, date_col, beacon_cols


def block_split(df, loc_col):
    """Within each location, earliest rows -> train, latest rows -> test."""
    rank = df.groupby(loc_col).cumcount()
    size = df.groupby(loc_col)[loc_col].transform("size")

    n_test = (size * TEST_FRAC).round().astype(int)
    n_test = n_test.where(size >= MIN_ROWS, 0)

    is_test = rank >= (size - n_test)
    is_gap = (rank >= (size - n_test - GAP)) & ~is_test & (n_test > 0)

    # Group id for CV: consecutive rows of one location share a group
    groups = df[loc_col] + "_" + (rank // BLOCK_SIZE).astype(str)
    return ~is_test & ~is_gap, is_test, groups


def to_xy(label):
    """'O07' -> (14, 7): letter = column, number = row (grid units)."""
    return ord(label[0].upper()) - ord("A"), int(label[1:])


def grid_error(y_true, y_pred):
    errs, near = [], []
    for t, p in zip(y_true, y_pred):
        (tx, ty), (px, py) = to_xy(t), to_xy(p)
        errs.append(np.hypot(tx - px, ty - py))
        near.append(max(abs(tx - px), abs(ty - py)) <= 1)
    return np.array(errs), np.array(near)


def main():
    df, loc_col, date_col, beacon_cols = load(CSV_PATH)
    print(f"Rows: {len(df)}, locations: {df[loc_col].nunique()}, "
          f"beacons: {len(beacon_cols)}")
    print(f"Date range: {df[date_col].min()} -> {df[date_col].max()}")
    days_per_loc = df.groupby(loc_col)[date_col].apply(lambda s: s.dt.date.nunique())
    print("Days each location was recorded on:")
    print(days_per_loc.value_counts().sort_index().to_string(), "\n")

    X = df[beacon_cols].replace(-200, MISSING_FILL)
    y = df[loc_col]

    train_mask, test_mask, groups = block_split(df, loc_col)
    X_train, y_train, g_train = X[train_mask], y[train_mask], groups[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    print(f"Train rows: {len(X_train)}, test rows: {len(X_test)}\n")

    models = {
        "Decision tree": (
            DecisionTreeClassifier(random_state=SEED),
            {"max_depth": [5, 10, 20, None]},
        ),
        "kNN": (
            KNeighborsClassifier(),
            {"n_neighbors": [1, 3, 5, 9], "weights": ["uniform", "distance"]},
        ),
        "Random forest": (
            RandomForestClassifier(random_state=SEED, n_jobs=-1),
            {"n_estimators": [100, 300], "max_depth": [10, 20, None]},
        ),
    }

    # Stratified + grouped: whole blocks are held out, but every location
    # still has blocks left in the training part of each fold.
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    best_name, best_score, best_model = None, -1, None

    for name, (est, grid) in models.items():
        search = GridSearchCV(est, grid, cv=cv, scoring="accuracy", n_jobs=-1)
        search.fit(X_train, y_train, groups=g_train)
        print(f"{name}: best CV accuracy {search.best_score_:.3f} "
              f"with {search.best_params_}")
        if search.best_score_ > best_score:
            best_name, best_score, best_model = name, search.best_score_, search.best_estimator_

    # Final evaluation on the held-out test blocks: do this once
    print(f"\nBest model by validation: {best_name}")
    pred = best_model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, pred):.3f}\n")
    joblib.dump(best_model, "best_model.joblib")
    errs, near = grid_error(y_test.tolist(), pred.tolist())
    print(f"Mean error: {errs.mean():.2f} grid cells")
    print(f"Median error: {np.median(errs):.2f} grid cells")
    print(f"Within 1 cell (incl. diagonals): {near.mean():.1%}\n")

    print(classification_report(y_test, pred, zero_division=0))


if __name__ == "__main__":
    main()
