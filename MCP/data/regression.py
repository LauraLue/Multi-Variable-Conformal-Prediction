"""Regression benchmarks CASP and SCM20D: data loading and neural-network predictors.

The datasets are not part of the repository; download them (see README) to
MCP/data/regression_CASP.csv and MCP/data/regression_scm20d.csv.  For each
dataset, train_*_nn() fits an MLP point predictor on n_train rows (CASP: the
first rows; SCM20D: randomly drawn rows) and returns the remaining rows as a
pool from which calibration and test data are drawn.

Preprocessing:
  CASP    targets RMSD and F7, features F1-F6, F8, F9, as in Feldman et al.
          (2023); all-NaN feature columns and rows with NaN are removed.  The
          rows are already in random order.
  SCM20D  the first 61 columns are features (column 0, headed "@data" by the
          ARFF conversion, is the time unit), the last 16 columns outputs, of
          which the first output_dim are used; rows with NaN/inf are removed
          and the rest shuffled before the training split, since the file
          consists of blocks of consecutive days of separate simulation runs.
For both datasets, features and targets are min-max normalized with the
ranges of the training rows, so that the pool does not enter the
preprocessing.
"""

import os

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def find_dataset(name: str) -> tuple[str, str]:
    """Locate a dataset file by name and infer its task type.

    Args:
        name: Substring to match against filenames in DATA_DIR.

    Returns:
        Tuple of (filepath, task_type) where task_type is "classification" or
        "regression".

    Raises:
        FileNotFoundError: If no matching file is found in DATA_DIR.
    """
    files = os.listdir(DATA_DIR)
    matches = [f for f in files if name.lower() in f.lower() and f.endswith((".csv", ".xlsx"))]
    if not matches:
        raise FileNotFoundError(f"No dataset matching '{name}' in {DATA_DIR}.\nAvailable: {files}")
    if len(matches) > 1:
        exact = [
            f
            for f in matches
            if f.lower().startswith(("classification_", "regression_"))
            and name.lower() in f.lower()
        ]
        if exact:
            matches = exact[:1]
        else:
            matches = matches[:1]
    fname = matches[0]
    task = "classification" if fname.startswith("classification") else "regression"
    return os.path.join(DATA_DIR, fname), task


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load a CSV or Excel file, auto-detecting whether it has a header row.

    Args:
        filepath: Absolute path to the dataset file (.csv or .xlsx).

    Returns:
        Raw DataFrame.
    """
    if filepath.endswith(".xlsx"):
        return pd.read_excel(filepath)
    with open(filepath) as fh:
        first_line = fh.readline().strip()
    try:
        [float(v) for v in first_line.split(",")]
        return pd.read_csv(filepath, header=None)
    except ValueError:
        return pd.read_csv(filepath)


def minmax_normalize(train: np.ndarray, pool: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Min-max normalize train and pool column-wise with the ranges of train.

    Columns that are constant on train are only shifted.  Pool values may
    fall outside [0, 1].
    """
    col_min = np.nanmin(train, axis=0)
    rng = np.nanmax(train, axis=0) - col_min
    rng[rng == 0] = 1
    return (train - col_min) / rng, (pool - col_min) / rng


def prepare_casp(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess CASP with the two-dimensional response of Feldman et al. (2023).

    The raw columns are RMSD, F1, ..., F9.  As in Feldman et al. (2023,
    App. B.2, data set "bio"), the targets are the original response RMSD
    and the feature F7, and the features are F1-F6, F8, F9.  Drop all-NaN
    feature columns and rows with any NaN.  The data is not normalized here
    (see train_casp_nn).

    Args:
        df: Raw CASP DataFrame.

    Returns:
        Tuple of (X, y) as numpy arrays; y holds (RMSD, F7).
    """
    n_y = 2

    # Positional: [F1..F6, F8, F9] features, then the targets [RMSD, F7]
    reordered = df.iloc[:, [1, 2, 3, 4, 5, 6, 8, 9, 0, 7]].reset_index(drop=True)

    data = reordered.values.astype(float)

    feat_all_nan = np.all(np.isnan(data[:, :-n_y]), axis=0)
    keep_cols = np.concatenate([~feat_all_nan, np.ones(n_y, dtype=bool)])
    data = data[:, keep_cols]
    data = data[~np.any(np.isnan(data), axis=1)]

    X = data[:, :-n_y]
    y = data[:, -n_y:]
    return X, y


def train_casp_nn(
    n_train: int,
    layers: list[int],
    epochs: int = 200,
    lr: float = 1e-3,
    random_state: int = 42,
) -> tuple:
    """Train an MLP on the first n_train CASP samples.

    Features and targets are min-max normalized with the ranges of the
    training rows.

    Args:
        n_train: Number of leading rows used for training.
        layers: Hidden layer sizes.
        epochs: Maximum training epochs.
        lr: Initial learning rate.
        random_state: Random seed.

    Returns:
        Tuple of (model, X_pool, y_pool) where X_pool/y_pool are the holdout
        rows beyond n_train available for experiment sampling.
    """
    filepath, _ = find_dataset("CASP")
    df = load_dataset(filepath)
    X, y = prepare_casp(df)

    X_train, X_pool = minmax_normalize(X[:n_train], X[n_train:])
    y_train, y_pool = minmax_normalize(y[:n_train], y[n_train:])

    print(
        f"Training CASP NNpredictor on {len(X_train)} samples, "
        f"layers {tuple(layers)}, pool size {len(X_pool)}"
    )

    model = MLPRegressor(
        hidden_layer_sizes=tuple(layers),
        max_iter=epochs,
        learning_rate_init=lr,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=False,
    )
    model.fit(X_train, y_train)
    print(f"  → stopped after {model.n_iter_} epochs")
    return model, X_pool, y_pool


def train_scm20d_nn(
    n_train: int,
    layers: list[int],
    output_dim: int = 2,
    epochs: int = 200,
    lr: float = 1e-3,
    random_state: int = 42,
) -> tuple:
    """Train an MLP on n_train randomly drawn SCM20D samples.

    SCM20D has 61 features and 16 output dimensions. This function selects
    the first ``output_dim`` outputs and applies min-max normalisation (same
    as CASP).  The rows are shuffled before the split: the file consists of
    blocks of consecutive days of separate simulation runs, so the leading
    rows would cover only the first runs.

    Args:
        n_train: Number of rows used for training.
        layers: Hidden layer sizes.
        output_dim: Number of output dimensions to use (2–16).
        epochs: Maximum training epochs.
        lr: Initial learning rate.
        random_state: Random seed of the shuffle and the MLP.

    Returns:
        Tuple of (model, X_pool, y_pool) where X_pool/y_pool are the holdout
        rows not used for training, available for experiment sampling.

    Raises:
        ValueError: If output_dim is outside [2, 16] or n_train >= cleaned
            dataset length.
    """
    if not 2 <= output_dim <= 16:
        raise ValueError(f"output_dim must be between 2 and 16, got {output_dim}")

    filepath, _ = find_dataset("SCM20D")
    df = load_dataset(filepath)

    # SCM20D: the first 61 columns are features (column 0, headed "@data" by
    # the ARFF conversion, is the time unit), the last 16 columns are outputs.
    X = df.iloc[:, :61].apply(pd.to_numeric, errors="coerce").values.astype(float)
    y_all = df.iloc[:, -16:].apply(pd.to_numeric, errors="coerce").values.astype(float)
    y = y_all[:, :output_dim]

    finite_mask = np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)
    n_removed = int((~finite_mask).sum())
    if n_removed > 0:
        print(f"SCM20D: removed {n_removed} rows with missing/NaN/inf values")
    X = X[finite_mask]
    y = y[finite_mask]

    if n_train >= len(X):
        raise ValueError(
            f"Not enough clean SCM20D rows after filtering: {len(X)} available, "
            f"but n_train={n_train} requires at least {n_train + 1}."
        )

    perm = np.random.default_rng(random_state).permutation(len(X))
    X, y = X[perm], y[perm]
    X_train, X_pool = minmax_normalize(X[:n_train], X[n_train:])
    y_train, y_pool = minmax_normalize(y[:n_train], y[n_train:])

    print(
        f"Training SCM20D NNpredictor on {len(X_train)} samples, "
        f"layers {tuple(layers)}, output_dim {output_dim}, pool size {len(X_pool)}"
    )

    model = MLPRegressor(
        hidden_layer_sizes=tuple(layers),
        max_iter=epochs,
        learning_rate_init=lr,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=False,
    )
    model.fit(X_train, y_train)
    print(f"  → stopped after {model.n_iter_} epochs")
    return model, X_pool, y_pool
