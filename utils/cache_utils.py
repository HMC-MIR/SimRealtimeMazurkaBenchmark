import os
import pickle
import hashlib
from typing import Any, Optional


def generate_cache_key(data: Any) -> Optional[str]:
    """
    Generate a unique cache key for the given data (e.g., numpy array or bytes).
    Returns None if data is None.
    """
    if data is None:
        return None
    if hasattr(data, "tobytes"):
        data_bytes = data.tobytes()
    elif isinstance(data, bytes):
        data_bytes = data
    else:
        raise ValueError("Data must be a numpy array or bytes-like object.")
    return hashlib.md5(data_bytes).hexdigest()


def load_cache(cache_dir: str, cache_key: str, suffix: str) -> Any:
    """
    Load cached data from disk given a directory, cache key, and suffix (e.g., 'chroma_stft_features.pkl').
    Returns None if not found.
    """
    if cache_dir is None or cache_key is None:
        return None
    cache_path = os.path.join(cache_dir, f"{cache_key}_{suffix}")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return None


def store_cache(cache_dir: str, cache_key: str, data: Any, suffix: str) -> None:
    """
    Save data to cache on disk given a directory, cache key, and suffix.
    """
    if cache_dir is None or cache_key is None:
        return
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{cache_key}_{suffix}")
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
