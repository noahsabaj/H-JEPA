import hashlib
import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import stable_worldmodel as swm
import torch


NORMALIZER_ARTIFACT_FILENAME = "normalizer.pt"
IMAGENET_STATS = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
MIN_STD = 1e-6  # a column with a smaller std (constant) is centred, not scaled


def save_atomic(obj, path) -> Path:
    """torch.save to <path>.tmp, then rename: a reader never sees a partial file at `path`."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
    return path


def write_text_atomic(path, text: str) -> Path:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
    return path


def file_fingerprint(path, block: int = 1 << 20) -> dict:
    """Cheap identity of a large file: its size and the sha256 of its first and last `block` bytes."""
    path = Path(path)
    size = path.stat().st_size
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(block))
        if size > block:
            f.seek(max(block, size - block))
            h.update(f.read(block))
    return {"size": size, "head_tail_sha256": h.hexdigest()}


def safe_std(std):
    """The z-score scale of a column: its std, or 1 where the column is constant. Training and
    planning both normalize with (x - mean) / safe_std(std)."""
    if torch.is_tensor(std):
        return torch.where(std > MIN_STD, std, torch.ones_like(std))
    std = np.asarray(std)
    return np.where(std > MIN_STD, std, 1).astype(std.dtype)


def check_stats(col: str, mean, std) -> None:
    for name, v in (("mean", mean), ("std", std)):
        if not np.isfinite(np.asarray(v, dtype=np.float64)).all():
            raise ValueError(f"Normalizer statistics of column {col!r}: {name} is not finite ({v})")


class ZScore:
    """(x - mean) / safe_std(std); NaN rows (padding) become 0 unless fill_nan=False. A class, not a
    closure, so dataset transforms pickle into spawned DataLoader workers."""

    def __init__(self, mean, std, fill_nan: bool = True):
        self.mean, self.std, self.fill_nan = mean, safe_std(std), fill_nan

    def __call__(self, x):
        x = (x - self.mean) / self.std
        if self.fill_nan:
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x.float()


class ColumnTransform:
    """Apply `fn` to `sample[source]` and store the result in `sample[target]`."""

    def __init__(self, fn, source: str, target: str):
        self.fn, self.source, self.target = fn, source, target

    def __call__(self, sample: dict) -> dict:
        sample[self.target] = self.fn(sample[self.source])
        return sample


class Compose:
    """Apply dict-sample transforms in sequence."""

    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


def build_hdf5_dataset(dataset_cfg, cache_dir=None):
    if OmegaConf.is_config(dataset_cfg):
        cfg = OmegaConf.to_container(dataset_cfg, resolve=True)
    else:
        cfg = dict(dataset_cfg)
    cfg.pop("val_name", None)
    if cfg.pop("type", None) == "droid":
        from droid_data import DROIDDataset

        return DROIDDataset(**cfg)
    subset_seed = int(cfg.pop("subset_seed", 0))
    total_transitions = cfg.pop("total_transitions", None)
    dataset = swm.data.HDF5Dataset(**cfg, cache_dir=cache_dir)
    if total_transitions is not None:
        episodes = _select_episodes_for_transition_budget(
            dataset, int(total_transitions), np.random.default_rng(subset_seed)
        )
        selected = set(episodes.tolist())
        dataset.clip_indices = [
            (episode, start)
            for episode, start in dataset.clip_indices
            if episode in selected
        ]
    return dataset


def _select_episodes_for_transition_budget(
    dataset,
    target_transitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    valid_episodes = np.flatnonzero(dataset.lengths >= dataset.last_level["span"])
    if valid_episodes.size == 0:
        raise ValueError(f"No valid episodes in {dataset.h5_path}.")

    available_transitions = int(dataset.lengths[valid_episodes].sum())
    if target_transitions > available_transitions:
        raise ValueError(
            f"Requested {target_transitions} transitions from {dataset.h5_path}, "
            f"but only {available_transitions} valid episode transitions are available."
        )

    shuffled = rng.permutation(valid_episodes)
    chosen = []
    total = 0

    for episode in shuffled:
        length = int(dataset.lengths[episode])
        previous_total = total
        chosen.append(int(episode))
        total += length

        if total >= target_transitions:
            if (
                len(chosen) > 1
                and abs(target_transitions - previous_total)
                < abs(target_transitions - total)
            ):
                chosen.pop()
            break

    if not chosen:
        chosen.append(int(shuffled[0]))

    return np.asarray(sorted(chosen), dtype=np.int64)


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    if hasattr(dataset, "get_col_stats"):
        mean_np, std_np = dataset.get_col_stats(source)
        mean = torch.from_numpy(np.array(mean_np)).float()
        std = torch.from_numpy(np.array(std_np)).float()
    else:
        col_data = dataset.get_col_data(source)
        data = torch.from_numpy(np.array(col_data))
        data = data[~torch.isnan(data).any(dim=1)]
        mean = data.mean(0, keepdim=True).clone()
        std = data.std(0, keepdim=True).clone()
    check_stats(source, mean, std)

    return ColumnTransform(ZScore(mean, std), source=source, target=target)


def get_column_normalizer_from_artifact(
    artifact: dict, source: str, target: str, fill_nan: bool = True
):
    """Get a column normalizer from saved training-set stats.

    With fill_nan=False, NaN rows are preserved so callers can mask them (used
    for probe targets whose padded/inactive slots are stored as NaN).
    """
    stats = artifact.get("stats", {})
    if source not in stats:
        available = ", ".join(sorted(stats.keys()))
        raise KeyError(
            f"Normalizer artifact is missing column {source!r}. "
            f"Available columns: {available}"
        )

    col_stats = stats[source]
    mean = torch.from_numpy(np.asarray(col_stats["mean"])).float()
    std = torch.from_numpy(np.asarray(col_stats["std"])).float()
    return ColumnTransform(ZScore(mean, std, fill_nan), source=source, target=target)


def normalizer_columns_from_dataset_cfg(dataset_cfg) -> list[str]:
    """Return non-pixel dataset columns that need z-score normalization."""
    columns = []
    for key in ("keys_to_load", "keys_to_cache"):
        for col in dataset_cfg.get(key, []) or []:
            if col.startswith("pixels"):
                continue
            if col not in columns:
                columns.append(col)

    keys_to_merge = dataset_cfg.get("keys_to_merge", None)
    if keys_to_merge is not None:
        for col in keys_to_merge.keys():
            if col.startswith("pixels"):
                continue
            if col not in columns:
                columns.append(col)

    return columns


def _column_mean_std(dataset, col: str) -> tuple[np.ndarray, np.ndarray, int]:
    if hasattr(dataset, "get_col_stats"):
        mean, std = dataset.get_col_stats(col)
        count = int(dataset.lengths.sum())
        return (
            np.asarray(mean, dtype=np.float64),
            np.asarray(std, dtype=np.float64),
            count,
        )

    data = np.asarray(dataset.get_col_data(col))
    flat = data.reshape(data.shape[0], -1)
    valid_mask = ~np.isnan(flat).any(axis=1)
    valid = data[valid_mask].astype(np.float64, copy=False)
    count = int(valid.shape[0])
    if count == 0:
        raise ValueError(f"Column {col!r} has no finite rows.")

    mean = valid.mean(axis=0, keepdims=True)
    std = valid.std(axis=0, ddof=1 if count > 1 else 0, keepdims=True)
    return mean, std, count


def build_normalizer_artifact(cfg, train_dataset) -> dict:
    """Build serializable training normalizer stats for planning eval."""
    dataset_cfg = cfg.data.dataset
    stats = {}
    for col in normalizer_columns_from_dataset_cfg(dataset_cfg):
        mean, std, count = _column_mean_std(train_dataset, col)
        check_stats(col, mean, std)
        stats[col] = {
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "count": count,
        }

    return {
        "format": "lejepa_training_normalizer_v1",
        "stats": stats,
        "metadata": {
            "dataset_name": str(dataset_cfg.get("name", "")),
            "columns": list(stats.keys()),
            "std_ddof": 1,
        },
    }


def save_normalizer_artifact(artifact: dict, run_dir: str | Path) -> Path:
    return save_atomic(artifact, Path(run_dir) / NORMALIZER_ARTIFACT_FILENAME)


def normalizer_stats_sha256(artifact: dict) -> str:
    """Hash of a normalizer artifact's statistics (mean, std, count of every column)."""
    h = hashlib.sha256()
    for col in sorted(artifact["stats"]):
        s = artifact["stats"][col]
        h.update(col.encode())
        for k in ("mean", "std"):
            h.update(np.ascontiguousarray(np.asarray(s[k], dtype=np.float32)).tobytes())
        h.update(str(int(s.get("count", 0))).encode())
    return h.hexdigest()


def load_normalizer_artifact(path: str | Path) -> dict:
    artifact = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict):
        raise TypeError(f"Normalizer artifact must be a dict, got {type(artifact)}.")
    if artifact.get("format") != "lejepa_training_normalizer_v1":
        raise ValueError(
            f"Unsupported normalizer artifact format: {artifact.get('format')!r}."
        )
    if not isinstance(artifact.get("stats"), dict):
        raise ValueError("Normalizer artifact is missing a 'stats' dictionary.")
    for col, s in artifact["stats"].items():
        check_stats(col, s["mean"], s["std"])
    return artifact
