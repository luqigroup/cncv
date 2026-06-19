"""Automatic download of the released CNCV datasets from a public mirror.

The Darcy and Helmholtz joint training datasets are too large to track in git,
so they are hosted publicly and fetched on first use. Each entry below maps a
dataset file name to a direct-download URL; :func:`ensure_dataset` downloads the
file the first time a script asks for it and is a no-op afterwards.

A user therefore does not need to regenerate the data (which requires the
external PDE solvers): the figure and evaluation scripts call
:func:`ensure_dataset` on their ``--data_path`` before loading it.
"""

from __future__ import annotations

import os
import sys
import urllib.request

# Direct-download URLs for the released datasets, keyed by file name.
# These are public links to the files under MyDropbox:cncv/data/ (see README).
DATASET_URLS: dict[str, str] = {
    "darcy_K10_N131072_grid40.npz":
        "https://www.dropbox.com/scl/fi/71zo8uoyvy3xnsfgjqzvd/"
        "darcy_K10_N131072_grid40.npz?rlkey=zlzerd1j0mukltb25jhxh10jm&dl=1",
    "helmholtz_K10_N131072_grid32.npz":
        "https://www.dropbox.com/scl/fi/catpkkyqv6rce56uwcge8/"
        "helmholtz_K10_N131072_grid32.npz?rlkey=qibi038a3zllw9a8zt1u5ivyr&dl=1",
}


def _progress_hook(block_idx: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    done = min(block_idx * block_size, total_size)
    pct = 100.0 * done / total_size
    sys.stdout.write(f"\r[cncv] downloading: {done / 1e6:6.0f} / "
                     f"{total_size / 1e6:.0f} MB ({pct:5.1f}%)")
    sys.stdout.flush()
    if done >= total_size:
        sys.stdout.write("\n")


def ensure_dataset(path: str) -> str:
    """Return ``path``, downloading the dataset from the public mirror if absent.

    Args:
        path: Where the dataset is expected, e.g.
            ``data/darcy/darcy_K10_N131072_grid40.npz``. May be absolute or
            relative. If the file is missing and its file name is a known
            release dataset, it is downloaded to ``path``.

    Returns:
        ``path`` unchanged, after ensuring the file exists locally.

    Raises:
        RuntimeError: if the file is missing and no download URL is registered
            for its name, or the URL is still a placeholder.
    """
    if os.path.exists(path):
        return path

    name = os.path.basename(path)
    url = DATASET_URLS.get(name)
    if url is None:
        raise RuntimeError(
            f"{path!r} is missing and no download URL is registered for "
            f"{name!r}. Generate it with the matching scripts/generate_*.py, "
            f"or add its URL to cncv/utils/download.py.")
    if url.endswith("_PLACEHOLDER"):
        raise RuntimeError(
            f"The download URL for {name!r} has not been configured. See the "
            f"dataset section of the README.")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"[cncv] dataset {name} not found locally; fetching from the public "
          f"mirror (this happens once)...")
    tmp = path + ".part"
    try:
        urllib.request.urlretrieve(url, tmp, _progress_hook)
        # A .npz is a zip archive; a wrong/expired link returns HTML instead.
        with open(tmp, "rb") as f:
            magic = f.read(4)
        if magic != b"PK\x03\x04":
            raise RuntimeError(
                f"the download for {name!r} did not return a valid .npz file "
                f"(the public link may have changed); see the README.")
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    os.replace(tmp, path)
    print(f"[cncv] saved {path} ({os.path.getsize(path) / 1e6:.0f} MB)")
    return path
