"""
Smoke tests for the CASP and SCM20D loaders and predictors on small synthetic
files with the layout of the real datasets (which are not in the repository).
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

import MCP.data.regression as regression


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(regression, "DATA_DIR", str(tmp_path))
    return tmp_path


def _write_casp(path, n=300, seed=0):
    rng = np.random.default_rng(seed)
    columns = ["RMSD"] + [f"F{i}" for i in range(1, 10)]
    pd.DataFrame(rng.standard_normal((n, 10)), columns=columns).to_csv(path, index=False)


def _write_scm20d(path, n=300, seed=1):
    rng = np.random.default_rng(seed)
    columns = ["@data"] + [f"x{i}" for i in range(60)] + [f"y{i}" for i in range(16)]
    pd.DataFrame(rng.standard_normal((n, 77)), columns=columns).to_csv(path, index=False)


def test_find_dataset_matches_name_and_task(data_dir):
    _write_casp(data_dir / "regression_CASP.csv")
    path, task = regression.find_dataset("casp")
    assert path.endswith("regression_CASP.csv") and task == "regression"
    with pytest.raises(FileNotFoundError, match="No dataset matching 'SCM20D'"):
        regression.find_dataset("SCM20D")


def test_prepare_casp_reorders_without_normalizing(data_dir):
    _write_casp(data_dir / "regression_CASP.csv")
    df = regression.load_dataset(str(data_dir / "regression_CASP.csv"))
    X, y = regression.prepare_casp(df)
    assert X.shape == (300, 8) and y.shape == (300, 2)
    # As in Feldman et al. (2023): targets RMSD and F7, features F1-F6, F8, F9
    np.testing.assert_array_equal(y, df[["RMSD", "F7"]].to_numpy())
    np.testing.assert_array_equal(X, df[["F1", "F2", "F3", "F4", "F5", "F6", "F8", "F9"]])


def test_minmax_normalize_uses_training_ranges():
    train = np.array([[0.0, 5.0], [2.0, 5.0], [4.0, 5.0]])
    pool = np.array([[6.0, 7.0], [-2.0, 5.0]])
    train_n, pool_n = regression.minmax_normalize(train, pool)
    np.testing.assert_allclose(train_n, [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    # Pool values outside the training range leave [0, 1]; constant columns are only shifted
    np.testing.assert_allclose(pool_n, [[1.5, 2.0], [-0.5, 0.0]])


def test_train_casp_normalizes_with_training_rows(data_dir):
    path = data_dir / "regression_CASP.csv"
    _write_casp(path)
    df = pd.read_csv(path)
    df.loc[250:, "RMSD"] += 100.0  # pool targets far outside the training range
    df.to_csv(path, index=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        _, _, y_pool = regression.train_casp_nn(n_train=200, layers=[8], epochs=5)
    assert y_pool[50:, 0].min() > 1.0  # scaled with the training range, not the pooled one


@pytest.mark.parametrize("dataset", ["CASP", "SCM20D"])
def test_train_nn_returns_model_and_holdout_pool(data_dir, dataset):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if dataset == "CASP":
            _write_casp(data_dir / "regression_CASP.csv")
            model, X_pool, y_pool = regression.train_casp_nn(n_train=200, layers=[8], epochs=20)
            n_out = 2
        else:
            _write_scm20d(data_dir / "regression_scm20d.csv")
            model, X_pool, y_pool = regression.train_scm20d_nn(
                n_train=200, layers=[8], output_dim=3, epochs=20
            )
            n_out = 3
    assert X_pool.shape[0] == y_pool.shape[0] == 100 and y_pool.shape[1] == n_out
    assert model.predict(X_pool).shape == (100, n_out)


def test_train_scm20d_uses_all_features_and_shuffles(data_dir):
    path = data_dir / "regression_scm20d.csv"
    _write_scm20d(path)
    # Mark each row in the time-unit column and append the blank trailing row
    # of the real file, which must be dropped.
    df = pd.read_csv(path)
    df["@data"] = np.arange(len(df), dtype=float)
    df.to_csv(path, index=False)
    with open(path, "a") as fh:
        fh.write("  " + "," * 76 + "\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model, X_pool, _ = regression.train_scm20d_nn(
            n_train=200, layers=[8], output_dim=2, epochs=5
        )
    assert model.n_features_in_ == 61 and X_pool.shape == (100, 61)
    # The pool is a random subset of the rows, not the trailing ones: had the
    # model been trained on rows 0..199, the normalized time unit of the pool
    # (rows 200..299) would exceed 1 everywhere.
    assert np.mean(X_pool[:, 0] <= 1.0) > 0.9


def test_train_scm20d_rejects_invalid_output_dim(data_dir):
    with pytest.raises(ValueError, match="output_dim"):
        regression.train_scm20d_nn(n_train=10, layers=[4], output_dim=17)
