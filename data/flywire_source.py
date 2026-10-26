"""
data/flywire_source.py
======================

REAL connectome support: streams the public **FlyWire FAFB v783** wiring
diagram and turns it into the `ConnectomeGraph` this project simulates.

Why this file exists
--------------------
The project previously only *documented* how to export a real edge list from
neuPrint/FlyWire and hand it to ``--connectome-path``.  This module makes
real connectivity work out of the box, with no account or API token:

* connectivity: FlyWire FAFB v783 proofread connections, Zenodo record
  ``10.5281/zenodo.10676866`` (CC-BY-4.0), 852 MB feather.  It contains
  ~16.85 M ``(pre, post, neuropil)`` synapse-count rows.
* annotations: Schlegel et al. 2024 Supplementary Data 5 (Nature,
  ``10.1038/s41586-024-07686-5``), 27 MB TSV.  It contains 139,255
  proofread neurons with cell type, predicted neurotransmitter, side and
  soma coordinates.

Both files are downloaded once into a local cache and verified against
published sizes/checksums.  Nothing is bundled in the repository.

Evidence labels (see docs/SIMULATION_NOTES.md)
----------------------------------------------
CONNECTOME - everything read from those two sources: neuron count, cell
             types, predicted neurotransmitter, soma coordinates, synapse
             counts between identified neurons.
SIMULATION - ``log1p`` weight compression, per-neuron incoming-weight
             normalisation, and the leaky-rate dynamics in
             ``brain/neuron_model.py``.
ASSUMPTION - the interactive 20k-neuron subset selection; mapping soma
             positions onto the retina sampling grid; treating the retina
             model's contrast output as an effective excitatory drive even
             though real photoreceptors release histamine (inhibitory).
             Each of these is called out in the code below.

What "real" means here, precisely
---------------------------------
The *connectivity structure* is measured.  The *dynamics* that run on it
are still the project's generic rate model, not fly electrophysiology.  This
is the same philosophy as the flyvis model (Lappalainen et al. 2024):
wire the circuit the way the real circuit is wired, then run simple dynamics
on that structure.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy import sparse

from brain.connectome import ConnectomeGraph, _normalize_incoming_weights
from brain.regions import OTHER_REGION, REGION_ORDER

# ---------------------------------------------------------------------------
# Public data sources
# ---------------------------------------------------------------------------
ANNOTATIONS_URL = (
    "https://media.springernature.com/original/springer-static/esm/"
    "art%3A10.1038%2Fs41586-024-07686-5/MediaObjects/"
    "41586_2024_7686_MOESM5_ESM.tsv"
)
ANNOTATIONS_SIZE = 27_015_208
ANNOTATIONS_SHA256 = (
    "30be6c73975a70c56d930e27911f36455d3886e15abf383b78edd2a5d679e0b6"
)
CONNECTIVITY_URL = (
    "https://zenodo.org/api/records/10676866/files/"
    "proofread_connections_783.feather/content"
)
CONNECTIVITY_SIZE = 852_022_274
CONNECTIVITY_MD5 = "f48f972d262323a102aed49af1396b8a"

DOI_CONNECTIVITY = "10.5281/zenodo.10676866"
DOI_ANNOTATIONS = "10.1038/s41586-024-07686-5"
LICENSE = "CC-BY-4.0"

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "real" / "flywire_v783"
ANNOTATIONS_FILENAME = "annotations.tsv"
CONNECTIVITY_FILENAME = "proofread_connections_783.feather"

# ---------------------------------------------------------------------------
# Subset selection / graph construction defaults
# ---------------------------------------------------------------------------
# Bump this when region mapping / selection semantics change, so cached
# graphs are rebuilt instead of silently mixing conventions.
SELECTION_VERSION = 1

DEFAULT_MAX_NEURONS = 20_000
DEFAULT_MIN_SYNAPSES_SUBSET = 1
DEFAULT_MIN_SYNAPSES_FULL = 5
# SIMULATION: cap on each neuron's total absolute incoming weight.  The raw
# log1p sum for a real neuron is typically much larger; 1.5 keeps the generic
# rate model in a responsive, non-saturated regime on real FlyWire graphs
# (2.5 made the mushroom-body loop saturate to ~1.0).
DEFAULT_WEIGHT_TARGET = 1.5

OPTIC_LOBE_REGIONS = ("retina", "lamina", "medulla", "lobula", "lobula_plate")
NAMED_REGIONS = OPTIC_LOBE_REGIONS + ("central_complex", "mushroom_body")
TILE_GRID_OPTIC = 16
TILE_GRID_CENTRAL = 6

# FAFB is a 4 x 4 x 40 nm volume; the annotation coordinates are in voxels.
# CONNECTOME/ASSUMPTION: this affine conversion is used only for the 3D
# layout and the retinotopic column mapping, never for connectivity.
VOXEL_NM = np.array([4.0, 4.0, 40.0], dtype=np.float64)

# Predicted neurotransmitter -> effective sign in the rate model.
# BIOLOGY: acetylcholine is the main fast excitatory transmitter in the fly
# brain; GABA and glutamate are inhibitory (adult Drosophila glutamate acts
# largely through inhibitory GluCl channels); histamine is the photoreceptor
# transmitter and is inhibitory at the lamina.
# ASSUMPTION: dopamine/serotonin/octopamine are treated as weak modulatory
# excitation (0.25) rather than modelled as neuromodulators.
SIGN_BY_NT = {
    "acetylcholine": 1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "histamine": -1.0,
    "glycine": -1.0,
    "dopamine": 0.25,
    "serotonin": 0.25,
    "octopamine": 0.25,
}
DEFAULT_SIGN = 1.0

PHOTORECEPTOR_TYPES = ("R1-6", "R7", "R8")

# Keep at least a few of the cell types that make the visual pathway
# interpretable when the user asks for a very small subset.
PRIORITY_CELL_TYPES = (
    "R1-6", "R7", "R8",
    "L1", "L2", "L3", "L4", "L5",
    "T4a", "T4b", "T4c", "T4d",
    "T5a", "T5b", "T5c", "T5d",
)
PRIORITY_MIN_PER_TYPE = 3


@dataclass(frozen=True)
class FlyWireFiles:
    annotations: Path
    connectivity: Path


# ---------------------------------------------------------------------------
# Download / cache management
# ---------------------------------------------------------------------------
def ensure_flywire_files(
    cache_dir: str | os.PathLike | None = None,
    *,
    allow_download: bool = True,
    verbose: bool = True,
) -> FlyWireFiles:
    """Return local paths to the two FlyWire files, downloading if needed.

    Downloads are cached in ``cache_dir`` (default ``data/real/flywire_v783``)
    and verified against the published size/checksum.  Set ``allow_download``
    to False for offline runs: a clear error is raised if the cache is empty.
    """
    cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)

    annotations = cache / ANNOTATIONS_FILENAME
    connectivity = cache / CONNECTIVITY_FILENAME

    _ensure_one_file(
        annotations,
        url=ANNOTATIONS_URL,
        expected_size=ANNOTATIONS_SIZE,
        checksum=ANNOTATIONS_SHA256,
        algorithm="sha256",
        label="Schlegel et al. 2024 annotations",
        allow_download=allow_download,
        verbose=verbose,
    )
    _ensure_one_file(
        connectivity,
        url=CONNECTIVITY_URL,
        expected_size=CONNECTIVITY_SIZE,
        checksum=CONNECTIVITY_MD5,
        algorithm="md5",
        label="FlyWire FAFB v783 connectivity",
        allow_download=allow_download,
        verbose=verbose,
    )

    provenance = cache / "PROVENANCE.json"
    if not provenance.exists():
        _write_provenance(provenance)

    return FlyWireFiles(annotations=annotations, connectivity=connectivity)


def _ensure_one_file(
    path: Path,
    *,
    url: str,
    expected_size: int,
    checksum: str,
    algorithm: str,
    label: str,
    allow_download: bool,
    verbose: bool,
) -> None:
    marker = path.with_name(path.name + ".verified")
    if path.exists() and path.stat().st_size == expected_size:
        if marker.exists() and marker.read_text().strip() == checksum:
            return
        # File present but never verified (e.g. copied into the cache).
        if _hash_file(path, algorithm) == checksum:
            marker.write_text(checksum + "\n")
            return
        if not allow_download:
            raise RuntimeError(
                f"{label} at {path} has an unexpected checksum. Delete it and "
                f"re-run with downloads enabled, or pass a different --connectome-cache."
            )
        if verbose:
            print(f"[flywire] checksum mismatch for {path}; re-downloading", flush=True)
        path.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)

    if not allow_download:
        raise FileNotFoundError(
            f"{label} is not cached at {path}.\n"
            f"Run once with downloads enabled (default), or download it manually:\n"
            f"  {url}\n"
            f"and place it at {path}"
        )

    _download_file(path, url=url, expected_size=expected_size, verbose=verbose)
    actual = _hash_file(path, algorithm)
    if actual != checksum:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded {label} failed {algorithm} verification "
            f"(expected {checksum}, got {actual}). Please retry."
        )
    marker.write_text(checksum + "\n")
    if verbose:
        print(f"[flywire] verified {path.name} ({expected_size / 1e6:.0f} MB)", flush=True)


def _download_file(path: Path, *, url: str, expected_size: int, verbose: bool) -> None:
    part = path.with_name(path.name + ".part")
    part.unlink(missing_ok=True)
    if verbose:
        print(
            f"[flywire] downloading {path.name} "
            f"({expected_size / 1e6:.0f} MB) -> {path}",
            flush=True,
        )
    req = urllib.request.Request(url, headers={"User-Agent": "fly_brain_sim/1.0"})
    downloaded = 0
    last_report = -1
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as response, open(part, "wb") as fh:
            total = int(response.headers.get("Content-Length") or expected_size)
            while True:
                chunk = response.read(4 * 1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                downloaded += len(chunk)
                pct = int(downloaded * 100 / max(total, 1))
                if verbose and pct // 10 != last_report:
                    last_report = pct // 10
                    elapsed = max(time.time() - t0, 1e-6)
                    print(
                        f"[flywire]   {pct:3d}%  {downloaded / 1e6:6.0f}/"
                        f"{total / 1e6:.0f} MB  ({downloaded / elapsed / 1e6:.1f} MB/s)",
                        flush=True,
                    )
    except Exception:
        part.unlink(missing_ok=True)
        raise
    if downloaded != expected_size:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded {downloaded} bytes for {path.name}, expected {expected_size}."
        )
    os.replace(part, path)


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_provenance(path: Path) -> None:
    payload = {
        "dataset": "FlyWire FAFB v783 (proofread wiring diagram)",
        "license": LICENSE,
        "connectivity": {
            "file": CONNECTIVITY_FILENAME,
            "url": CONNECTIVITY_URL,
            "doi": DOI_CONNECTIVITY,
            "size": CONNECTIVITY_SIZE,
            "md5": CONNECTIVITY_MD5,
        },
        "annotations": {
            "file": ANNOTATIONS_FILENAME,
            "url": ANNOTATIONS_URL,
            "doi": DOI_ANNOTATIONS,
            "size": ANNOTATIONS_SIZE,
            "sha256": ANNOTATIONS_SHA256,
        },
        "citations": [
            "Dorkenwald et al. 2024, Nature, doi:10.1038/s41586-024-07558-y",
            "Schlegel et al. 2024, Nature, doi:10.1038/s41586-024-07686-5",
        ],
        "fetched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Annotation parsing / region and neurotransmitter mapping
# ---------------------------------------------------------------------------
_ANNOTATION_COLUMNS = [
    "root_id",
    "super_class",
    "cell_class",
    "cell_type",
    "flow",
    "side",
    "top_nt",
    "known_nt",
    "soma_x",
    "soma_y",
    "soma_z",
    "pos_x",
    "pos_y",
    "pos_z",
]


def load_annotations(path: str | os.PathLike, *, verbose: bool = False) -> pd.DataFrame:
    """Read the Schlegel et al. annotation table and add project fields.

    Returns a DataFrame with one row per proofread neuron and these added
    columns: ``region``, ``nt``, ``sign`` (effective sign used by the rate
    model), and physical-ish ``x``/``y``/``z`` coordinates.
    """
    t0 = time.time()
    df = pd.read_csv(path, sep="\t", low_memory=False, usecols=_ANNOTATION_COLUMNS)
    if df["root_id"].duplicated().any():
        raise ValueError("annotation table contains duplicate root_id values")
    df["root_id"] = df["root_id"].astype(np.int64)
    df["region"] = map_regions(df)
    df["nt"] = primary_neurotransmitter(df)
    df["sign"] = effective_sign(df)
    xyz = physical_coordinates(df)
    df["x"], df["y"], df["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if verbose:
        print(
            f"[flywire] loaded {len(df):,} annotated neurons in {time.time() - t0:.1f}s",
            flush=True,
        )
    return df


def map_regions(df: pd.DataFrame) -> np.ndarray:
    """Map FlyWire cell classes/types to this project's coarse brain regions.

    CONNECTOME: the cell_class strings (ME, LA, LO, LOP, CX, Kenyon_Cell, ...)
    are the published Schlegel et al. annotations.  ASSUMPTION: the collapse
    to seven coarse regions, and assigning T4/T5 to ``lobula_plate`` because
    that is where their axons terminate and where the motion readout lives.
    """
    cc = df["cell_class"].fillna("").astype(str)
    ct = df["cell_type"].fillna("").astype(str)

    region = np.full(len(df), OTHER_REGION, dtype=object)
    # Order matters: LOP must be checked before LO.
    region[cc.str.startswith("ME").to_numpy()] = "medulla"
    region[cc.str.startswith("LA").to_numpy()] = "lamina"
    region[cc.str.startswith("LOP").to_numpy()] = "lobula_plate"
    region[(cc.str.startswith("LO") & ~cc.str.startswith("LOP")).to_numpy()] = "lobula"
    region[cc.str.startswith("CX").to_numpy()] = "central_complex"
    region[cc.str.startswith(("Kenyon", "MBON", "DAN", "APL")).to_numpy()] = "mushroom_body"
    # Photoreceptors are annotated as cell_class "visual" with these types.
    region[ct.isin(PHOTORECEPTOR_TYPES).to_numpy()] = "retina"
    # T4/T5 somata sit in the medulla/lobula cortex, axons in the lobula plate.
    region[ct.str.startswith(("T4", "T5")).to_numpy()] = "lobula_plate"
    region[ct.str.startswith("APL").to_numpy()] = "mushroom_body"
    return region


def primary_neurotransmitter(df: pd.DataFrame) -> np.ndarray:
    """Return one predicted transmitter string per neuron.

    Uses the curated ``known_nt`` field when present, otherwise the
    connectome-wide ``top_nt`` prediction.  BIOLOGY: photoreceptors release
    histamine; the per-neuron prediction is noisy for R1-R6/R7, so their
    identity is overridden with the known transmitter.
    """
    known = (
        df["known_nt"].fillna("").astype(str).str.split(",").str[0].str.strip().str.lower()
    )
    top = df["top_nt"].fillna("").astype(str).str.strip().str.lower()
    nt = np.where(known.to_numpy() != "", known.to_numpy(), top.to_numpy())
    # Normalise a few spelling variants used in the source table.
    replacements = {"hist": "histamine", "5-ht": "serotonin", "5ht": "serotonin"}
    nt = np.array([replacements.get(str(x), str(x)) for x in nt], dtype=object)
    ct = df["cell_type"].fillna("").astype(str)
    nt[ct.isin(PHOTORECEPTOR_TYPES).to_numpy()] = "histamine"
    return nt


def effective_sign(df: pd.DataFrame) -> np.ndarray:
    """Effective pre-synaptic sign used by the rate network.

    CONNECTOME: transmitter identity is predicted/known per neuron.
    ASSUMPTION: our retina model outputs a contrast/luminance drive rather
    than a histamine release rate, so photoreceptor output is treated as
    excitatory here even though their real transmitter is inhibitory.  Without
    this, a bright video frame would inhibit the lamina and silence the
    pathway in this rate-model abstraction.  The original ``nt`` field is kept
    for transparency.
    """
    nt = df["nt"].to_numpy()
    sign = np.array([SIGN_BY_NT.get(str(x), DEFAULT_SIGN) for x in nt], dtype=np.float32)
    photos = df["cell_type"].fillna("").astype(str).isin(PHOTORECEPTOR_TYPES).to_numpy()
    sign[photos] = 1.0
    return sign


def physical_coordinates(df: pd.DataFrame) -> np.ndarray:
    """Physically proportioned [x, y, z] coordinates (nm-ish) for each neuron.

    Uses the soma position when annotated, otherwise the synapse/position
    centroid.  FAFB coordinates are in 4 x 4 x 40 nm voxel units, so the
    axes are scaled accordingly.  SIMULATION/ASSUMPTION: this conversion
    affects the 3D layout and column mapping only; connectivity is untouched.
    """
    def pick(soma: str, pos: str) -> np.ndarray:
        s = df[soma].to_numpy(dtype=np.float64)
        p = df[pos].to_numpy(dtype=np.float64)
        return np.where(np.isfinite(s), s, p)

    x = pick("soma_x", "pos_x") * VOXEL_NM[0]
    y = pick("soma_y", "pos_y") * VOXEL_NM[1]
    z = pick("soma_z", "pos_z") * VOXEL_NM[2]
    return np.stack([x, y, z], axis=1)


# ---------------------------------------------------------------------------
# Subset selection
# ---------------------------------------------------------------------------
def select_neurons(
    df: pd.DataFrame,
    max_neurons: int,
    *,
    seed: int = 0,
    verbose: bool = False,
) -> np.ndarray:
    """Select a connectivity-preserving subset of real neurons.

    ``max_neurons <= 0`` returns every annotated neuron (whole brain).
    Otherwise a proportional quota per region is filled by selecting
    retinotopic *tiles* (a shared set of 16x16 (y, z) tiles per hemisphere
    for the optic lobe, separate small tile grids for CX/MB).  Selecting
    whole tiles rather than isolated neurons keeps local columnar circuits
    intact, which random subsampling destroys.

    ASSUMPTION: this is a display/compute-budget choice, not a biological
    claim.  The full graph is available with ``max_neurons=0``.
    """
    rng = np.random.default_rng(seed)
    named_mask = (df["region"] != OTHER_REGION).to_numpy()
    n_named = int(named_mask.sum())

    if max_neurons is None or max_neurons <= 0 or max_neurons >= n_named:
        if max_neurons is not None and 0 < max_neurons < n_named:
            # Quota exceeded for a selection request; still honour a cap by
            # selecting all named regions rather than silently including
            # unclassified neurons.
            return np.flatnonzero(named_mask)
        return np.arange(len(df))

    counts = df.loc[named_mask, "region"].value_counts()
    total_named = int(counts.sum())
    quotas = {
        region: max(1, int(round(max_neurons * int(count) / total_named)))
        for region, count in counts.items()
    }
    # Correct tiny rounding drift on the largest region.
    drift = max_neurons - sum(quotas.values())
    if drift:
        largest = max(quotas, key=quotas.get)
        quotas[largest] = max(1, quotas[largest] + drift)

    sides = df["side"].fillna("na").astype(str).to_numpy()
    region = df["region"].to_numpy()
    selected = np.zeros(len(df), dtype=bool)

    # --- optic lobe: one shared tile set per hemisphere -----------------
    optic_mask = np.isin(region, OPTIC_LOBE_REGIONS)
    optic_quota = sum(quotas[r] for r in OPTIC_LOBE_REGIONS if r in quotas)
    optic_total = int((named_mask & optic_mask).sum())
    for side in _side_groups(sides[optic_mask]):
        mask = named_mask & optic_mask & (sides == side)
        if not mask.any():
            continue
        target = int(round(optic_quota * mask.sum() / max(optic_total, 1)))
        selected |= _select_tiles(df, mask, target, TILE_GRID_OPTIC, rng)

    # --- central complex / mushroom body: their own small tile grids ----
    for region_name in ("central_complex", "mushroom_body"):
        if region_name not in quotas:
            continue
        region_mask = named_mask & (region == region_name)
        if not region_mask.any():
            continue
        for side in _side_groups(sides[region_mask]):
            mask = region_mask & (sides == side)
            target = int(round(quotas[region_name] * mask.sum() / max(region_mask.sum(), 1)))
            selected |= _select_tiles(df, mask, target, TILE_GRID_CENTRAL, rng)

    # --- make sure key visual cell types exist even in tiny subsets -----
    selected |= _ensure_priority_types(df, selected, rng)

    n_selected = int(selected.sum())
    if verbose:
        print(
            f"[flywire] selected {n_selected:,} neurons "
            f"(requested {max_neurons:,}, seed {seed})",
            flush=True,
        )
    return np.flatnonzero(selected)


def _side_groups(sides: np.ndarray) -> list[str]:
    groups = [s for s in ("left", "right") if np.any(sides == s)]
    if not groups:
        groups = sorted(set(sides.tolist()))
    return groups


def _select_tiles(
    df: pd.DataFrame,
    mask: np.ndarray,
    target: int,
    n_tiles: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select complete (y, z) tiles from ``mask`` until ``target`` is reached."""
    empty = np.zeros(len(df), dtype=bool)
    idx = np.flatnonzero(mask)
    if target <= 0 or idx.size == 0:
        return empty
    if target >= idx.size:
        return mask.copy()

    y = df["y"].to_numpy(dtype=np.float64)[idx]
    z = df["z"].to_numpy(dtype=np.float64)[idx]
    y0, y1 = np.percentile(y, [1, 99])
    z0, z1 = np.percentile(z, [1, 99])
    if not np.isfinite(y0) or y1 <= y0:
        y0, y1 = float(np.min(y)), float(np.max(y) + 1.0)
    if not np.isfinite(z0) or z1 <= z0:
        z0, z1 = float(np.min(z)), float(np.max(z) + 1.0)

    ty = np.clip(((y - y0) / (y1 - y0) * n_tiles).astype(np.int64), 0, n_tiles - 1)
    tz = np.clip(((z - z0) / (z1 - z0) * n_tiles).astype(np.int64), 0, n_tiles - 1)
    tile = ty * n_tiles + tz
    counts = np.bincount(tile, minlength=n_tiles * n_tiles)

    order = rng.permutation(np.flatnonzero(counts))
    chosen: list[int] = []
    total = 0
    for t in order:
        if total >= target:
            break
        chosen.append(int(t))
        total += int(counts[t])

    selected = np.zeros(len(df), dtype=bool)
    selected[idx[np.isin(tile, chosen)]] = True
    return selected


def _ensure_priority_types(
    df: pd.DataFrame,
    selected: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add a few neurons of key visual types if the tile sample missed them."""
    extra = np.zeros(len(df), dtype=bool)
    cell_type = df["cell_type"].fillna("").astype(str).to_numpy()
    for name in PRIORITY_CELL_TYPES:
        candidates = np.flatnonzero((cell_type == name) & ~selected)
        if candidates.size == 0:
            continue
        if not selected[cell_type == name].any():
            pick = rng.choice(candidates, size=min(PRIORITY_MIN_PER_TYPE, candidates.size), replace=False)
            extra[pick] = True
    return extra


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_flywire_connectome(
    config,
    *,
    cache_dir: str | os.PathLike | None = None,
    annotations_path: str | os.PathLike | None = None,
    connectivity_path: str | os.PathLike | None = None,
    max_neurons: int | None = None,
    min_synapses: int | None = None,
    allow_download: bool | None = None,
    use_cache: bool = True,
    rebuild: bool = False,
    verbose: bool = True,
) -> ConnectomeGraph:
    """Build a `ConnectomeGraph` from the real FlyWire FAFB v783 dataset.

    Parameters
    ----------
    config : SimulationConfig
        Supplies ``random_seed``, ``retina_grid_shape``, and defaults for
        ``max_neurons`` / ``min_synapses`` when not passed explicitly.
    cache_dir : path, optional
        Where the downloaded files and the built-graph cache live.
    annotations_path, connectivity_path : path, optional
        Use already-downloaded files directly (mainly for tests/offline use).
    max_neurons : int, optional
        ``0`` = whole brain (139,255 neurons); ``None`` uses config/default
        (20,000 for interactive use).  Positive values request approximately
        that many neurons from the seven named regions.
    min_synapses : int, optional
        Minimum *summed* synapses per directed pair.  Defaults to 1 for a
        subset and 5 for the whole brain (the published FAFB threshold).
    """
    t_start = time.time()
    if allow_download is None:
        allow_download = bool(getattr(config, "allow_connectome_download", True))

    if annotations_path is not None and connectivity_path is not None:
        files = FlyWireFiles(Path(annotations_path), Path(connectivity_path))
    else:
        files = ensure_flywire_files(
            cache_dir=cache_dir or getattr(config, "connectome_cache_dir", None),
            allow_download=allow_download,
            verbose=verbose,
        )

    if max_neurons is None:
        configured = getattr(config, "max_neurons", None)
        max_neurons = int(configured) if configured is not None else DEFAULT_MAX_NEURONS
    if min_synapses is None:
        configured = getattr(config, "min_synapses", None)
        if configured is not None:
            min_synapses = int(configured)
        else:
            min_synapses = (
                DEFAULT_MIN_SYNAPSES_FULL if max_neurons <= 0 else DEFAULT_MIN_SYNAPSES_SUBSET
            )

    seed = int(getattr(config, "random_seed", 0))
    grid_shape = tuple(int(v) for v in config.retina_grid_shape)
    weight_target = float(DEFAULT_WEIGHT_TARGET)

    cache_path = _cache_path(
        cache_dir=cache_dir or getattr(config, "connectome_cache_dir", None) or files.connectivity.parent,
        max_neurons=max_neurons,
        min_synapses=min_synapses,
        seed=seed,
        grid_shape=grid_shape,
        weight_target=weight_target,
    )
    if use_cache and not rebuild and cache_path.exists():
        cached = _load_graph_cache(
            cache_path,
            config=config,
            files=files,
            max_neurons=max_neurons,
            min_synapses=min_synapses,
            seed=seed,
            grid_shape=grid_shape,
            weight_target=weight_target,
            verbose=verbose,
        )
        if cached is not None:
            return cached

    # ---- annotations + subset -----------------------------------------
    ann = load_annotations(files.annotations, verbose=verbose)
    if max_neurons > 0:
        chosen = select_neurons(ann, max_neurons, seed=seed, verbose=verbose)
    else:
        chosen = np.arange(len(ann))
    ann = ann.iloc[chosen].reset_index(drop=True)
    # Stable edge-id ordering for searchsorted mapping.
    ann = ann.sort_values("root_id", kind="stable").reset_index(drop=True)
    ids = ann["root_id"].to_numpy(dtype=np.int64)
    n = len(ids)
    if n == 0:
        raise ValueError("FlyWire selection produced zero neurons")

    # ---- stream the 16.8M (pre, post, neuropil) rows --------------------
    t0 = time.time()
    pre_chunks: list[np.ndarray] = []
    post_chunks: list[np.ndarray] = []
    syn_chunks: list[np.ndarray] = []
    total_rows = 0
    kept_rows = 0
    with pa.memory_map(str(files.connectivity), "rb") as source:
        reader = pa.ipc.open_file(source)
        names = reader.schema.names
        try:
            pre_i = names.index("pre_pt_root_id")
            post_i = names.index("post_pt_root_id")
            syn_i = names.index("syn_count")
        except ValueError as exc:
            raise ValueError(
                "Unrecognized FlyWire connectivity schema. Expected columns "
                "pre_pt_root_id, post_pt_root_id, syn_count."
            ) from exc

        n_batches = reader.num_record_batches
        for batch_i in range(n_batches):
            batch = reader.get_batch(batch_i)
            pre = batch.column(pre_i).to_numpy()
            post = batch.column(post_i).to_numpy()
            syn = batch.column(syn_i).to_numpy()
            total_rows += pre.size

            pre_pos = np.searchsorted(ids, pre)
            valid_pre = pre_pos < n
            # Guard the gather against out-of-range indices before comparing.
            valid_pre[valid_pre] = ids[pre_pos[valid_pre]] == pre[valid_pre]
            post_pos = np.searchsorted(ids, post)
            valid_post = post_pos < n
            valid_post[valid_post] = ids[post_pos[valid_post]] == post[valid_post]
            keep = valid_pre & valid_post
            if keep.any():
                pre_chunks.append(pre_pos[keep].astype(np.int32))
                post_chunks.append(post_pos[keep].astype(np.int32))
                syn_chunks.append(syn[keep].astype(np.float32))
                kept_rows += int(keep.sum())

            if verbose and (batch_i % 32 == 0 or batch_i == n_batches - 1):
                pct = (batch_i + 1) * 100 / max(n_batches, 1)
                print(
                    f"[flywire]   streamed {pct:5.1f}%  "
                    f"{total_rows:,} rows, {kept_rows:,} touching selection",
                    flush=True,
                )

    if not pre_chunks:
        raise ValueError(
            "No edges connected the selected neurons. Increase --max-neurons "
            "or choose a different selection seed."
        )
    pre = np.concatenate(pre_chunks)
    post = np.concatenate(post_chunks)
    syn = np.concatenate(syn_chunks)
    del pre_chunks, post_chunks, syn_chunks

    adjacency = sparse.coo_matrix((syn, (pre, post)), shape=(n, n)).tocsr()
    del pre, post, syn
    adjacency.sum_duplicates()

    if min_synapses > 1:
        adjacency.data[adjacency.data < min_synapses] = 0.0
        adjacency.eliminate_zeros()

    if adjacency.nnz == 0:
        raise ValueError(
            f"No connections remained at the >= {min_synapses} synapse threshold."
        )

    # ---- weights and signs ---------------------------------------------
    # SIMULATION: compress the heavy-tailed synapse-count distribution.
    adjacency.data = np.log1p(adjacency.data).astype(np.float32, copy=False)
    sign = ann["sign"].to_numpy(dtype=np.float32)
    adjacency = sparse.diags(sign).dot(adjacency).tocsr()
    # SIMULATION: cap total absolute incoming drive per neuron so a handful
    # of huge fan-in neurons does not saturate the rate model.
    adjacency = _normalize_incoming_weights(adjacency, target_abs_sum=weight_target)

    # ---- contiguous region slices + scene layout -----------------------
    region_rank = {r: i for i, r in enumerate(REGION_ORDER + [OTHER_REGION])}
    region_values = ann["region"].to_numpy()
    order = np.argsort(
        np.fromiter((region_rank.get(r, len(region_rank)) for r in region_values), dtype=np.int64),
        kind="stable",
    )
    ann = ann.iloc[order].reset_index(drop=True)
    adjacency = adjacency[order][:, order]
    ids = ids[order]
    sign = sign[order]
    region_values = region_values[order]

    positions = _scene_positions(ann)
    column_of = _column_map(ann, grid_shape)

    region_slices: dict[str, slice] = {}
    start = 0
    for region_name in REGION_ORDER + [OTHER_REGION]:
        count = int((region_values == region_name).sum())
        if count:
            region_slices[region_name] = slice(start, start + count)
            start += count
    if start != n:
        raise RuntimeError("internal error: region slices do not cover all neurons")

    graph = ConnectomeGraph(
        n_neurons=n,
        adjacency=adjacency.tocsr(),
        region_of=np.asarray(region_values, dtype=object),
        region_slices=region_slices,
        positions=positions,
        column_of=column_of,
        source="flywire_fafb783",
        level=config.level.value,
        neuron_ids=ids.astype(np.int64),
        cell_type_of=ann["cell_type"].fillna("").astype(str).to_numpy(dtype=object),
        nt_of=ann["nt"].astype(str).to_numpy(dtype=object),
        sign_of=sign,
    )

    if verbose:
        print(
            f"[flywire] graph ready: {n:,} neurons, {adjacency.nnz:,} connections "
            f"(>= {min_synapses} syn), {time.time() - t_start:.1f}s total",
            flush=True,
        )

    if use_cache:
        _save_graph_cache(
            cache_path,
            graph,
            files=files,
            max_neurons=max_neurons,
            min_synapses=min_synapses,
            seed=seed,
            grid_shape=grid_shape,
            weight_target=weight_target,
        )
    return graph


def _scene_positions(df: pd.DataFrame) -> np.ndarray:
    """Real soma coordinates converted into the PyVista scene frame.

    Scene axes are [left-right, dorsal-ventral, anterior-posterior], i.e.
    the raw [x, z, y].  The whole cloud is centred and uniformly scaled to
    about +/-4 scene units so the existing camera works.
    """
    x = df["x"].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.float64)
    z = df["z"].to_numpy(dtype=np.float64)
    x = x - np.median(x)
    y = y - np.median(y)
    z = z - np.median(z)
    span = max(
        float(np.percentile(np.abs(x), 99)),
        float(np.percentile(np.abs(y), 99)),
        float(np.percentile(np.abs(z), 99)),
        1.0,
    )
    scene = np.stack([x, z, y], axis=1) / span * 4.0
    return scene.astype(np.float32)


def _column_map(df: pd.DataFrame, grid_shape: tuple[int, int]) -> np.ndarray:
    """Map optic-lobe neurons to a retinotopic column in the retina grid.

    ASSUMPTION: the fly optic lobe is retinotopic, so soma position in the
    (anterior-posterior, dorsal-ventral) plane is used as a proxy for visual
    field position.  Both hemispheres are mapped to the same grid because
    the simulation feeds the same video frame to both eyes.  This is a
    coarse visualization/drive mapping, not a measured retinotopic atlas.
    """
    rows, cols = grid_shape
    n = len(df)
    column = np.full(n, -1, dtype=np.int64)
    optic = df["region"].isin(OPTIC_LOBE_REGIONS).to_numpy()
    if not optic.any():
        return column

    y_all = df["y"].to_numpy(dtype=np.float64)
    z_all = df["z"].to_numpy(dtype=np.float64)
    sides = df["side"].fillna("na").astype(str).to_numpy()

    for side in _side_groups(sides[optic]):
        mask = optic & (sides == side)
        if not mask.any():
            continue
        y = y_all[mask]
        z = z_all[mask]
        y0, y1 = np.percentile(y, [1, 99])
        z0, z1 = np.percentile(z, [1, 99])
        if y1 <= y0:
            y1 = y0 + 1.0
        if z1 <= z0:
            z1 = z0 + 1.0
        u = np.clip((y - y0) / (y1 - y0), 0.0, 1.0)
        v = np.clip((z - z0) / (z1 - z0), 0.0, 1.0)
        row = np.minimum((v * rows).astype(np.int64), rows - 1)
        col = np.minimum((u * cols).astype(np.int64), cols - 1)
        column[mask] = row * cols + col
    return column


# ---------------------------------------------------------------------------
# Built-graph cache
# ---------------------------------------------------------------------------
def _cache_path(
    *,
    cache_dir: str | os.PathLike,
    max_neurons: int,
    min_synapses: int,
    seed: int,
    grid_shape: tuple[int, int],
    weight_target: float,
) -> Path:
    name = (
        f"graph_n{max_neurons}_s{min_synapses}_seed{seed}"
        f"_grid{grid_shape[0]}x{grid_shape[1]}_w{weight_target:g}"
        f"_v{SELECTION_VERSION}.npz"
    )
    return Path(cache_dir) / name


def _save_graph_cache(
    path: Path,
    graph: ConnectomeGraph,
    *,
    files: FlyWireFiles,
    max_neurons: int,
    min_synapses: int,
    seed: int,
    grid_shape: tuple[int, int],
    weight_target: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    region_cats, region_codes = _encode_categories(graph.region_of)
    cell_cats, cell_codes = _encode_categories(graph.cell_type_of)
    nt_cats, nt_codes = _encode_categories(graph.nt_of)
    slice_names = list(graph.region_slices.keys())
    slice_starts = np.array([graph.region_slices[r].start for r in slice_names], dtype=np.int64)
    slice_stops = np.array([graph.region_slices[r].stop for r in slice_names], dtype=np.int64)

    tmp = path.with_name(path.name + ".part.npz")
    np.savez_compressed(
        tmp,
        data=graph.adjacency.data,
        indices=graph.adjacency.indices,
        indptr=graph.adjacency.indptr,
        shape=np.array(graph.adjacency.shape, dtype=np.int64),
        region_codes=region_codes,
        region_categories=region_cats,
        celltype_codes=cell_codes,
        celltype_categories=cell_cats,
        nt_codes=nt_codes,
        nt_categories=nt_cats,
        positions=graph.positions,
        column_of=graph.column_of,
        neuron_ids=graph.neuron_ids,
        sign_of=graph.sign_of,
        slice_names=np.array(slice_names, dtype=str),
        slice_starts=slice_starts,
        slice_stops=slice_stops,
        source=np.array([graph.source], dtype=str),
        level=np.array([graph.level], dtype=str),
        selection_version=np.array(SELECTION_VERSION, dtype=np.int64),
        max_neurons=np.array(max_neurons, dtype=np.int64),
        min_synapses=np.array(min_synapses, dtype=np.int64),
        seed=np.array(seed, dtype=np.int64),
        grid_shape=np.array(grid_shape, dtype=np.int64),
        weight_target=np.array(weight_target, dtype=np.float64),
        connectivity_size=np.array(files.connectivity.stat().st_size, dtype=np.int64),
        connectivity_mtime_ns=np.array(files.connectivity.stat().st_mtime_ns, dtype=np.int64),
        annotations_size=np.array(files.annotations.stat().st_size, dtype=np.int64),
        annotations_mtime_ns=np.array(files.annotations.stat().st_mtime_ns, dtype=np.int64),
    )
    os.replace(tmp, path)


def _load_graph_cache(
    path: Path,
    *,
    config,
    files: FlyWireFiles,
    max_neurons: int,
    min_synapses: int,
    seed: int,
    grid_shape: tuple[int, int],
    weight_target: float,
    verbose: bool,
) -> ConnectomeGraph | None:
    try:
        z = np.load(path, allow_pickle=False)
    except Exception:
        return None
    try:
        def scalar(name):
            return np.asarray(z[name]).reshape(-1)[0]

        checks = [
            ("selection_version", SELECTION_VERSION),
            ("max_neurons", max_neurons),
            ("min_synapses", min_synapses),
            ("seed", seed),
            ("weight_target", weight_target),
            ("connectivity_size", files.connectivity.stat().st_size),
            ("connectivity_mtime_ns", files.connectivity.stat().st_mtime_ns),
            ("annotations_size", files.annotations.stat().st_size),
            ("annotations_mtime_ns", files.annotations.stat().st_mtime_ns),
        ]
        for name, expected in checks:
            if name not in z:
                return None
            if name == "weight_target":
                if not np.isclose(float(scalar(name)), float(expected)):
                    return None
            elif int(scalar(name)) != int(expected):
                return None
        if tuple(int(v) for v in np.asarray(z["grid_shape"]).tolist()) != grid_shape:
            return None

        shape = tuple(int(v) for v in np.asarray(z["shape"]).tolist())
        adjacency = sparse.csr_matrix(
            (z["data"], z["indices"], z["indptr"]), shape=shape
        ).tocsr()

        region_cats = [str(v) for v in np.asarray(z["region_categories"]).tolist()]
        region_of = np.array(
            [region_cats[int(i)] for i in z["region_codes"]], dtype=object
        )
        cell_cats = [str(v) for v in np.asarray(z["celltype_categories"]).tolist()]
        cell_type_of = np.array(
            [cell_cats[int(i)] for i in z["celltype_codes"]], dtype=object
        )
        nt_cats = [str(v) for v in np.asarray(z["nt_categories"]).tolist()]
        nt_of = np.array([nt_cats[int(i)] for i in z["nt_codes"]], dtype=object)

        names = [str(v) for v in np.asarray(z["slice_names"]).tolist()]
        starts = np.asarray(z["slice_starts"]).tolist()
        stops = np.asarray(z["slice_stops"]).tolist()
        region_slices = {name: slice(int(a), int(b)) for name, a, b in zip(names, starts, stops)}

        graph = ConnectomeGraph(
            n_neurons=shape[0],
            adjacency=adjacency,
            region_of=region_of,
            region_slices=region_slices,
            positions=np.asarray(z["positions"], dtype=np.float32),
            column_of=np.asarray(z["column_of"], dtype=np.int64),
            source=str(np.asarray(z["source"]).reshape(-1)[0]),
            level=str(np.asarray(z["level"]).reshape(-1)[0]),
            neuron_ids=np.asarray(z["neuron_ids"], dtype=np.int64),
            cell_type_of=cell_type_of,
            nt_of=nt_of,
            sign_of=np.asarray(z["sign_of"], dtype=np.float32),
        )
        if verbose:
            print(
                f"[flywire] loaded cached graph {path.name} "
                f"({shape[0]:,} neurons, {adjacency.nnz:,} connections)",
                flush=True,
            )
        return graph
    finally:
        z.close()


def _encode_categories(values) -> tuple[np.ndarray, np.ndarray]:
    if values is None:
        return np.array([], dtype=str), np.array([], dtype=np.int32)
    arr = np.asarray([str(v) for v in values], dtype=str)
    cats, codes = np.unique(arr, return_inverse=True)
    return cats, codes.astype(np.int32)


# ---------------------------------------------------------------------------
# Small CLI: pre-download the data
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Download the real FlyWire FAFB v783 connectome files."
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=f"Where to cache the ~880 MB of source files (default: {DEFAULT_CACHE_DIR}).",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Only check the cache; do not download missing files.",
    )
    args = parser.parse_args(argv)
    files = ensure_flywire_files(
        args.cache_dir, allow_download=not args.no_download, verbose=True
    )
    print(f"annotations:  {files.annotations}")
    print(f"connectivity: {files.connectivity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
