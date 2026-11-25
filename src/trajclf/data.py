"""Data extraction, preprocessing and splitting.

Raw data (from Inside, ``Project.zip``, kept in ``data/raw/`` and git-ignored):

* ``GBG500.parquet`` — 1,529,547 IMU readings, long format:
  ``ride_id, time_index, ax, ay, az, rx, ry, rz`` (1,807–4,767 readings per ride)
* ``GBG500_labels.csv`` — ``RIDE_ID, LABEL`` for the 500 rides
* ``SRV3490.parquet`` — crash ("topple") data, not used for the 5-class task

Data-quality decisions (see docs/FINDINGS.md):
* duplicate ``(ride_id, time_index)`` readings are averaged;
* missing sensor values (≈0.65 % of rows) are ignored by the aggregate
  statistics and linearly interpolated for the sequence model.

Three input representations, selected by ``data.mode`` in the config:

* ``aggregate`` — mean, std, min, max, median of each sensor = 30 features per
  ride. Identical to ``research/prepare_features.py`` (the team's first model).
* ``aggregate_plus`` — richer driving-style statistics (104 features): the 6 sensors
  plus the acceleration and rotation magnitudes, each summarised by mean, std, min,
  max, median, 5/25/75/95th percentiles, jerk (mean |Δ|, std Δ, max |Δ|) and the
  share of peaks beyond 2 std. Aimed at the overfitting seen with ``aggregate``.
* ``sequence`` — the whole ride resampled to ``seq_len`` steps × 6 sensors,
  for the 1D-CNN.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

STATS = ("mean", "std", "min", "max", "median")
PLUS_STATS = ("mean", "std", "min", "max", "median", "p05", "p25", "p75", "p95",
              "jerk_abs_mean", "jerk_std", "jerk_abs_max", "peak_rate")
AGGREGATE_MODES = ("aggregate", "aggregate_plus")


def _fill_nan(points: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaNs, column by column (edges: nearest value)."""
    out = points.copy()
    idx = np.arange(len(out))
    for j in range(out.shape[1]):
        col = out[:, j]
        bad = np.isnan(col)
        if bad.all():
            col[:] = 0.0
        elif bad.any():
            col[bad] = np.interp(idx[bad], idx[~bad], col[~bad])
    return out


def aggregate_features(points: np.ndarray) -> np.ndarray:
    """(T, F) -> (5F,) ordered sensor-major: ax_mean, ax_std, ..., rz_median.

    NaN-ignoring, std with ddof=1 — same as pandas ``agg`` in prepare_features.py.
    """
    with np.errstate(all="ignore"):
        stats = [np.nanmean(points, 0), np.nanstd(points, 0, ddof=1), np.nanmin(points, 0),
                 np.nanmax(points, 0), np.nanmedian(points, 0)]
    return np.nan_to_num(np.stack(stats, axis=1).reshape(-1)).astype(np.float32)


def plus_channels(feature_columns: list[str]) -> list[str]:
    """Sensors + magnitudes |a| (first 3 columns) and |r| (next 3) when 6 sensors are given."""
    return list(feature_columns) + (["acc_mag", "rot_mag"] if len(feature_columns) == 6 else [])


def aggregate_plus_features(points: np.ndarray) -> np.ndarray:
    """(T, F) -> (13 * C,) with C = F (+2 magnitudes if F == 6), ordered channel-major."""
    x = _fill_nan(np.asarray(points, dtype=np.float64))
    if x.shape[1] == 6:
        x = np.column_stack([x, np.linalg.norm(x[:, :3], axis=1), np.linalg.norm(x[:, 3:6], axis=1)])
    jerk = np.diff(x, axis=0) if len(x) > 1 else np.zeros((1, x.shape[1]))
    mean = x.mean(0)
    std = x.std(0, ddof=1) if len(x) > 1 else np.zeros(x.shape[1])
    p05, p25, med, p75, p95 = np.percentile(x, [5, 25, 50, 75, 95], axis=0)
    peak = (np.abs(x - mean) > 2 * np.maximum(std, 1e-9)).mean(0)
    stats = [mean, std, x.min(0), x.max(0), med, p05, p25, p75, p95,
             np.abs(jerk).mean(0), jerk.std(0), np.abs(jerk).max(0), peak]
    return np.nan_to_num(np.stack(stats, axis=1).reshape(-1)).astype(np.float32)


@dataclass
class Preprocessor:
    """Everything needed to turn one raw ride into a model input.

    Saved next to the model (``preprocess.json``) so that training, the API and
    the benchmarks apply exactly the same transformation.
    """

    feature_columns: list[str]
    labels: list[str]
    mode: str = "aggregate"           # "aggregate" | "aggregate_plus" | "sequence"
    seq_len: int = 1024
    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)

    @property
    def input_shape(self) -> tuple[int, ...]:
        n = len(self.feature_columns)
        if self.mode == "aggregate_plus":
            return (len(plus_channels(self.feature_columns)) * len(PLUS_STATS),)
        return (n * len(STATS),) if self.mode == "aggregate" else (self.seq_len, n)

    @property
    def input_names(self) -> list[str]:
        if self.mode == "aggregate":
            return [f"{c}_{s}" for c in self.feature_columns for s in STATS]
        if self.mode == "aggregate_plus":
            return [f"{c}_{s}" for c in plus_channels(self.feature_columns) for s in PLUS_STATS]
        return list(self.feature_columns)

    def _check(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != len(self.feature_columns):
            raise ValueError(f"expected shape (T, {len(self.feature_columns)}), got {points.shape}")
        if len(points) == 0:
            raise ValueError("empty ride")
        return points

    def resample(self, points: np.ndarray) -> np.ndarray:
        """Linearly resample a (T, F) ride to (seq_len, F), NaNs interpolated."""
        points = _fill_nan(self._check(points))
        if len(points) == 1:
            return np.repeat(points, self.seq_len, axis=0)
        src = np.linspace(0.0, 1.0, len(points))
        dst = np.linspace(0.0, 1.0, self.seq_len)
        return np.stack([np.interp(dst, src, points[:, j]) for j in range(points.shape[1])], 1).astype(np.float32)

    def raw_features(self, points: np.ndarray) -> np.ndarray:
        """Representation before standardisation."""
        if self.mode == "aggregate":
            return aggregate_features(self._check(points))
        if self.mode == "aggregate_plus":
            return aggregate_plus_features(self._check(points))
        return self.resample(points)

    def transform(self, points: np.ndarray) -> np.ndarray:
        x = self.raw_features(points)
        return ((x - np.asarray(self.mean, np.float32)) / np.asarray(self.std, np.float32)).astype(np.float32)

    def transform_batch(self, rides: list[np.ndarray]) -> np.ndarray:
        return np.stack([self.transform(r) for r in rides])

    def fit(self, rides: list[np.ndarray]) -> Preprocessor:
        """Fit standardisation on TRAINING rides only."""
        x = np.stack([self.raw_features(r) for r in rides])
        x = x.reshape(-1, x.shape[-1])  # aggregate: per feature; sequence: per sensor over time
        self.mean = x.mean(0).tolist()
        self.std = (x.std(0) + 1e-6).tolist()
        return self

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Preprocessor:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Raw data
# ---------------------------------------------------------------------------
def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def load_raw(cfg: dict) -> tuple[list[str], list[np.ndarray], np.ndarray]:
    """Return (ride ids, list of (T, F) arrays sorted by time, label strings)."""
    d = cfg["data"]
    feats, rid, tcol, lcol = d["feature_columns"], d["id_column"], d["time_column"], d["label_column"]
    df = _read(Path(d["raw_path"]))
    if d.get("labels_path"):
        lab = _read(Path(d["labels_path"])).rename(
            columns={d["labels_id_column"]: rid, d["labels_label_column"]: lcol})
        df = df.drop(columns=[lcol], errors="ignore")
    else:
        lab = df[[rid, lcol]].drop_duplicates(rid)
    missing = {rid, tcol, *feats} - set(df.columns)
    if missing:
        raise KeyError(f"columns missing from {d['raw_path']}: {sorted(missing)} — update configs/config.yaml")
    # Average duplicate (ride, time) readings, as in research/prepare_features.py
    df = df.groupby([rid, tcol], as_index=False)[feats].mean()
    labels_by_ride = dict(zip(lab[rid], lab[lcol].astype(str), strict=True))
    ids, rides, y = [], [], []
    for ride_id, g in df.groupby(rid, sort=True):  # sorted ride ids = same order as ride_features.csv
        if ride_id in labels_by_ride:
            ids.append(ride_id)
            rides.append(g[feats].to_numpy(np.float32))
            y.append(labels_by_ride[ride_id])
    return ids, rides, np.asarray(y)


def make_synthetic(cfg: dict, n_per_class: int = 40, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic IMU data with the configured columns -> (readings, labels).

    For tests / CI only. Each class has a different vibration amplitude and
    frequency, so metrics on this data say nothing about the real task.
    """
    rng = np.random.default_rng(seed)
    d = cfg["data"]
    frames, label_rows = [], []
    for cls, label in enumerate(cfg["project"]["labels"]):
        amp, freq = 1.0 + cls * 0.8, 0.5 + cls * 0.4
        for k in range(n_per_class):
            T = int(rng.integers(150, 400))
            t = np.arange(T) / 50.0
            ride = f"synthetic-{cls}-{k:03d}"
            block = {c: amp * np.sin(2 * np.pi * freq * t + rng.uniform(0, 6.28)) + rng.normal(0, 0.5, T)
                     for c in d["feature_columns"]}
            block[d["feature_columns"][0]][rng.integers(0, T)] = np.nan  # like the real data
            block[d["id_column"]] = ride
            block[d["time_column"]] = np.arange(T)
            frames.append(pd.DataFrame(block))
            label_rows.append({d.get("labels_id_column", d["id_column"]): ride,
                               d.get("labels_label_column", d["label_column"]): label})
    readings, labels = pd.concat(frames, ignore_index=True), pd.DataFrame(label_rows)
    if not d.get("labels_path"):
        readings = readings.merge(labels.set_axis([d["id_column"], d["label_column"]], axis=1))
    return readings, labels


def write_synthetic(cfg: dict, **kwargs) -> None:
    d = cfg["data"]
    readings, labels = make_synthetic(cfg, **kwargs)
    raw = Path(d["raw_path"])
    raw.parent.mkdir(parents=True, exist_ok=True)
    readings.to_parquet(raw, index=False) if raw.suffix == ".parquet" else readings.to_csv(raw, index=False)
    if d.get("labels_path"):
        labels.to_csv(d["labels_path"], index=False)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def build_datasets(cfg: dict) -> dict:
    """Load, split by ride (stratified), fit the preprocessor on train only.

    The test split reproduces research/train_baseline.py exactly
    (test_size=0.2, stratify, random_state=42, rides sorted by id, labels in
    alphabetical order), so results stay comparable with docs/FINDINGS.md.
    """
    d = cfg["data"]
    labels = cfg["project"]["labels"]
    ids, rides, y_str = load_raw(cfg)
    unknown = set(y_str) - set(labels)
    if unknown:
        raise ValueError(f"unknown labels {unknown}; update project.labels in config")
    y = np.asarray([labels.index(v) for v in y_str], dtype=np.int64)
    seed = cfg["project"]["seed"]
    idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=d["test_size"], stratify=y,
                                           random_state=seed)
    idx_val = np.asarray([], dtype=int)
    if d.get("val_size", 0) > 0:
        idx_train, idx_val = train_test_split(idx_train, test_size=d["val_size"] / (1 - d["test_size"]),
                                              stratify=y[idx_train], random_state=seed)

    pre = Preprocessor(d["feature_columns"], labels, d.get("mode", "aggregate"), d.get("seq_len", 1024))
    pre.fit([rides[i] for i in idx_train])

    def xs(sel):
        return pre.transform_batch([rides[i] for i in sel]) if len(sel) else np.zeros((0, *pre.input_shape),
                                                                                       np.float32)

    lengths = [len(r) for r in rides]
    return {
        "preprocessor": pre,
        "X_train": xs(idx_train), "y_train": y[idx_train],
        "X_val": xs(idx_val), "y_val": y[idx_val],
        "X_test": xs(idx_test), "y_test": y[idx_test],
        "test_ride_ids": np.asarray([ids[i] for i in idx_test]),
        "raw_stats": {
            "n_rides": len(y),
            "n_readings": int(sum(lengths)),
            "readings_per_ride": {"min": min(lengths), "median": int(np.median(lengths)), "max": max(lengths)},
            "raw_file_mb": round(Path(d["raw_path"]).stat().st_size / 1e6, 2),
            "class_counts": {labels[k]: int((y == k).sum()) for k in range(len(labels))},
            "input_shape": list(pre.input_shape),
        },
    }


def save_processed(ds: dict, out_dir: str | Path) -> Path:
    """Persist splits so that training and benchmarks use exactly the same data."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "splits.npz",
                        **{k: v for k, v in ds.items() if k.startswith(("X_", "y_", "test_ride"))})
    ds["preprocessor"].save(out / "preprocess.json")
    (out / "raw_stats.json").write_text(json.dumps(ds["raw_stats"], indent=2))
    return out


def main() -> None:
    """CLI: ``python -m trajclf.data [--synthetic]``."""
    import argparse

    from trajclf.config import load_config

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--synthetic", action="store_true", help="write synthetic raw files first (no real data)")
    p.add_argument("--config", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.synthetic:
        write_synthetic(cfg)
        print(f"synthetic data written to {cfg['data']['raw_path']}")
    ds = build_datasets(cfg)
    out = save_processed(ds, cfg["data"]["processed_dir"])
    print(json.dumps(ds["raw_stats"], indent=2))
    print(f"processed splits written to {out}")


if __name__ == "__main__":
    main()
