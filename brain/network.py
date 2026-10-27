"""
brain/network.py
=================

Ties together:
  vision.retina.RetinaModel       -> photoreceptor responses
  brain.motion_pathway.ReichardtMotionDetector -> local motion signals
  hearing.johnstons_organ.AuditoryReceptorModel -> auditory receptor responses
  brain.connectome.ConnectomeGraph -> wiring
  brain.neuron_model.LeakyRateNetwork -> dynamics

into a single `FlyBrain` object with a simple `step(frame_bgr) -> stats`
interface used by the simulation engine and the visualizers.

Data flow per video frame (this IS the real, documented Drosophila optic
pathway order; the specific numerical treatment at each stage is our
SIMULATION-level simplification - see the module docstrings referenced
below for the detailed biology/assumption breakdown of each stage):

    video frame
      -> RetinaModel (BIOLOGY: photoreceptors, ASSUMPTION: video as world)
      -> injected as external current into 'retina' neurons (SIMULATION)
      -> ReichardtMotionDetector on the lamina-stage grid (BIOLOGY-inspired
         T4/T5 motion detection, SIMULATION: HRC implementation)
      -> injected as extra external current into 'medulla' neurons
      -> full recurrent connectome propagation for N sub-steps (SIMULATION)
      -> downstream regions (lobula, lobula_plate, central_complex,
         mushroom_body) evolve from recurrent input alone, same as any
         other neuron in the graph.

    video soundtrack (ASSUMPTION: video audio as the acoustic world)
      -> AuditoryReceptorModel (BIOLOGY-inspired Johnston's organ / JO
         neurons, SIMULATION: FFT filterbank + adaptation)
      -> injected as external current into 'antenna' neurons in synthetic
         mode, or into real JO-* neurons in connectome mode
      -> propagates through the same recurrent dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from brain.connectome import ConnectomeGraph, build_synthetic_connectome, load_real_connectome
from brain.fly_state import EmotionTracker, FlyState, compute_fly_state
from brain.motion_pathway import ReichardtMotionDetector
from brain.neuron_model import LeakyRateNetwork
from brain.regions import OTHER_REGION, REGION_ORDER
from hearing.johnstons_organ import AuditoryReceptorModel
from simulation.config import SimLevel, SimulationConfig
from vision.retina import RetinaModel


# BIOLOGY: T4 (ON) and T5 (OFF) a-d are tuned to the four cardinal directions;
# Maisak et al. 2013 established a=front-to-back, b=back-to-front, c=up,
# d=down.  ASSUMPTION: the video's +x ("right") is identified with
# front-to-back; the absolute azimuth is arbitrary for a camera video.
T4_T5_DIRECTIONS = {
    "T4a": "right",
    "T5a": "right",
    "T4b": "left",
    "T5b": "left",
    "T4c": "up",
    "T5c": "up",
    "T4d": "down",
    "T5d": "down",
}
# SIMULATION: gain applied to the HRC direction maps before they are added to
# T4/T5 external drive.  Chosen so the motion signal is visible on the
# activation scale without swamping real connectome input.
MOTION_INJECTION_GAIN = 4.0
PHOTORECEPTOR_CHANNELS = ("R1-6", "R7", "R8")

# BIOLOGY: JO-* are the mechanosensory/auditory receptor neurons of
# Johnston's organ (Kamikouchi et al. 2006). ASSUMPTION: they all receive the
# same coarse band mixture; see hearing/johnstons_organ.py.
AUDITORY_CELL_TYPE_PREFIX = "JO-"
# SIMULATION: how strongly the auditory receptor response drives its target
# neurons. Chosen to be visible on the [0, 1] activity scale without
# swamping visual drive.
AUDIO_INJECTION_GAIN = 1.5
# ASSUMPTION: mid band (which contains the ~200 Hz courtship-song pulse
# frequency) weighted most, plus a transient/onset component.
AUDIO_BAND_WEIGHTS = {"low": 0.6, "mid": 1.0, "high": 0.8}
AUDIO_ONSET_WEIGHT = 1.2


@dataclass
class StepStats:
    sim_time_s: float
    active_neuron_count: int
    active_fraction: float
    region_activity: dict[str, float]
    region_active_count: dict[str, int]
    mean_firing_rate_hz: float
    strongest_connections: list[tuple[int, int, float]]
    motion_energy_mean: float
    frame_process_ms: float
    strongest_connection_labels: list[str] = field(default_factory=list)
    # Auditory pathway (Johnston's organ). Band levels are receptor-model
    # outputs in [0, 1]; zero when hearing is disabled or the video is silent.
    hearing_active: bool = False
    audio_band_levels: dict[str, float] = field(default_factory=dict)
    audio_rms: float = 0.0
    audio_onset: float = 0.0
    # Illustrative activity-derived "mood" readout; see brain/fly_state.py.
    # Never a claim that the simulation experiences anything.
    fly_state: FlyState | None = None
    # Rolling frequency of the hard mood labels over recent frames (fractions
    # in [0, 1], summing to 1).  See brain/fly_state.EmotionTracker.
    emotion_mix: dict[str, float] = field(default_factory=dict)
    emotion_mix_frames: int = 0


class FlyBrain:
    def __init__(self, config: SimulationConfig):
        self.config = config
        if config.level == SimLevel.CONNECTOME:
            source = config.connectome_source or (
                "csv" if config.connectome_path else "flywire"
            )
            if source == "csv":
                if not config.connectome_path:
                    raise ValueError(
                        "CONNECTOME level with --connectome-source csv requires "
                        "--connectome-path"
                    )
                self.graph: ConnectomeGraph = load_real_connectome(
                    config.connectome_path, config
                )
            elif source == "flywire":
                try:
                    from data.flywire_source import build_flywire_connectome
                except ImportError as exc:  # pragma: no cover - dependency error path
                    raise ImportError(
                        "FlyWire mode needs pyarrow (and pandas). Install the "
                        "requirements with: pip install -r requirements.txt"
                    ) from exc
                self.graph = build_flywire_connectome(
                    config,
                    use_cache=True,
                    rebuild=config.rebuild_connectome,
                )
            else:
                raise ValueError(
                    f"Unknown connectome source {source!r}; use 'csv' or 'flywire'"
                )
        else:
            self.graph = build_synthetic_connectome(config)

        self.retina_model = RetinaModel(config.retina_grid_shape)
        self.motion_detector = ReichardtMotionDetector(config.retina_grid_shape)
        self.auditory_model = AuditoryReceptorModel(config.audio_sample_rate)
        self.dynamics = LeakyRateNetwork(self.graph.adjacency, config.dynamics, seed=config.random_seed)
        self.emotion_tracker = EmotionTracker()

        # Real-connectome metadata: fast cell-type -> index lookups for
        # targeted visual drive (photoreceptors and T4/T5 subtypes) and
        # auditory drive (Johnston's-organ JO-* receptor neurons).
        self._cell_type_indices: dict[str, np.ndarray] = {}
        self._auditory_indices = np.empty(0, dtype=np.int64)
        if self.graph.cell_type_of is not None:
            cell_types = np.asarray(self.graph.cell_type_of, dtype=object)
            for name in PHOTORECEPTOR_CHANNELS + tuple(T4_T5_DIRECTIONS):
                self._cell_type_indices[name] = np.flatnonzero(cell_types == name)
            is_auditory = np.fromiter(
                (str(ct).startswith(AUDITORY_CELL_TYPE_PREFIX) for ct in cell_types),
                dtype=bool,
                count=cell_types.size,
            )
            self._auditory_indices = np.flatnonzero(is_auditory)
        self._is_real = self.graph.source != "synthetic"

        # Precompute the strongest synapses once; per-frame "strongest active
        # connections" then only has to scan this small candidate set instead
        # of the full connectome (important for 139k-neuron whole-brain mode).
        coo = self.graph.adjacency.tocoo()
        n_top = min(coo.nnz, 60_000)
        if n_top:
            top = np.argsort(-np.abs(coo.data))[:n_top]
            self._top_edge_src = coo.row[top].astype(np.int32)
            self._top_edge_dst = coo.col[top].astype(np.int32)
            self._top_edge_w = coo.data[top].astype(np.float32)
        else:
            self._top_edge_src = np.empty(0, dtype=np.int32)
            self._top_edge_dst = np.empty(0, dtype=np.int32)
            self._top_edge_w = np.empty(0, dtype=np.float32)

        self._sim_time_s = 0.0
        self._last_motion_energy = 0.0
        self._last_luminance_grid = None
        self._last_direction_maps = None
        self._last_audio_features = self._silent_audio_features()
        self._external_input = np.zeros(self.graph.n_neurons, dtype=np.float32)

    @staticmethod
    def _silent_audio_features() -> dict[str, float]:
        return {
            "low": 0.0,
            "mid": 0.0,
            "high": 0.0,
            "rms": 0.0,
            "onset": 0.0,
            "total": 0.0,
        }

    def reset(self):
        self.dynamics.reset()
        self._sim_time_s = 0.0
        self.retina_model._baseline = None
        self.auditory_model.reset()
        self._last_audio_features = self._silent_audio_features()
        self.emotion_tracker.reset()

    def _inject_vision(self, frame_bgr: np.ndarray, dt: float):
        vision = self.retina_model.process(frame_bgr)
        r1_6 = vision["r1_6"]
        r7_8 = vision["r7_8"]
        self._last_luminance_grid = vision["raw_luminance"]

        self._external_input[:] = 0.0

        # HRC motion detection is computed on the same downsampled luminance
        # grid regardless of which connectome drives the downstream brain.
        rows, cols_ = self.config.retina_grid_shape
        lamina_grid_est = r1_6.reshape(rows, cols_)
        direction_maps = self.motion_detector.step(lamina_grid_est, dt)
        self._last_direction_maps = direction_maps
        self._last_motion_energy = float(direction_maps["motion_energy"].mean())

        if self._is_real:
            self._inject_real_vision(vision, direction_maps)
            return

        # ---- synthetic pathway (DEMO / DETAILED) ------------------------
        retina_slice = self.graph.region_slices.get("retina")
        lamina_slice = self.graph.region_slices.get("lamina")
        medulla_slice = self.graph.region_slices.get("medulla")

        if retina_slice is not None:
            cols = self.graph.column_of[retina_slice]
            # map each retina neuron to its column's photoreceptor sample,
            # cycling if oversubscribed (n_neurons can exceed n_columns).
            grid_flat = r1_6
            n_grid = grid_flat.shape[0]
            sample = grid_flat[cols % n_grid]
            self._external_input[retina_slice] = sample * 2.0

        if medulla_slice is not None and self.config.motion_injection:
            cols_m = self.graph.column_of[medulla_slice]
            motion_flat = direction_maps["motion_energy"].flatten()
            n_grid = motion_flat.shape[0]
            sample = motion_flat[cols_m % n_grid]
            self._external_input[medulla_slice] += sample * 1.5

        if lamina_slice is not None:
            cols_l = self.graph.column_of[lamina_slice]
            n_grid = r7_8.shape[0]
            sample = r7_8[cols_l % n_grid]
            self._external_input[lamina_slice] += sample * 0.5

    def _inject_real_vision(self, vision: dict, direction_maps: dict):
        """Inject a video frame into a real connectome.

        Only identified photoreceptors (R1-6, R7/R8) and, optionally, the
        four T4/T5 direction subtypes are driven directly. Everything else -
        lamina, medulla, lobula, central complex, mushroom body - is driven
        through the measured FlyWire synapses.  If the CSV export has no cell
        types (or the selected subgraph contains none), fall back to the
        previous behaviour of uniform mean drive into the 'retina' region.
        """
        r1_6 = vision["r1_6"]
        r7_8 = vision["r7_8"]
        ct = self.graph.cell_type_of

        if ct is None or not self._cell_type_indices:
            retina_slice = self.graph.region_slices.get("retina")
            target = retina_slice if retina_slice is not None else slice(0, self.graph.n_neurons)
            self._external_input[target] = float(r1_6.mean()) * 2.0
            return

        # BIOLOGY: R1-R6 are the achromatic motion channel; R7/R8 are the
        # chromatic/UV channel.  Injected with the same gains as the synthetic
        # pathway so the two modes are visually comparable.
        self._inject_cell_types(("R1-6",), r1_6, gain=2.0)
        self._inject_cell_types(("R7", "R8"), r7_8, gain=0.5)

        if self.config.motion_injection:
            for cell_type, direction in T4_T5_DIRECTIONS.items():
                self._inject_cell_types(
                    (cell_type,),
                    direction_maps[direction].flatten(),
                    gain=MOTION_INJECTION_GAIN,
                )

        # If the selection contains no photoreceptors at all (tiny custom
        # CSV), still give the retina region a visual drive.
        has_photoreceptors = any(
            self._cell_type_indices.get(name, np.empty(0)).size > 0
            for name in PHOTORECEPTOR_CHANNELS
        )
        if not has_photoreceptors:
            retina_slice = self.graph.region_slices.get("retina")
            if retina_slice is not None:
                self._external_input[retina_slice] += float(r1_6.mean()) * 2.0

    def _inject_hearing(self, audio_window: np.ndarray | None):
        """Add auditory-receptor drive for the current video frame's sound.

        Target neurons: real ``JO-*`` Johnston's-organ neurons when the
        connectome has cell types, otherwise the ``antenna`` region (synthetic
        mode and CSV exports that label it). If neither exists, no neuron is
        driven directly; the stats still report the audio levels.
        """
        if (
            not self.config.hearing_enabled
            or audio_window is None
            or np.size(audio_window) == 0
        ):
            self._last_audio_features = self._silent_audio_features()
            return

        features = self.auditory_model.process(audio_window)
        self._last_audio_features = features

        drive = sum(
            AUDIO_BAND_WEIGHTS[name] * features.get(name, 0.0)
            for name in AUDIO_BAND_WEIGHTS
        )
        drive = (
            min(2.0, max(0.0, drive + AUDIO_ONSET_WEIGHT * features["onset"]))
            * AUDIO_INJECTION_GAIN
            * float(self.config.auditory_gain)
        )
        if drive <= 0.0:
            return

        if self._is_real:
            targets = self._auditory_indices
            if targets.size == 0:
                sl = self.graph.region_slices.get("antenna")
                if sl is not None and sl.stop > sl.start:
                    targets = np.arange(sl.start, sl.stop)
            if targets.size == 0:
                # Last resort: a CSV export with no cell types/antenna region.
                sl = self.graph.region_slices.get(OTHER_REGION)
                if sl is not None and sl.stop > sl.start:
                    targets = np.arange(sl.start, sl.stop)
            if targets.size:
                self._external_input[targets] += drive
            return

        antenna_slice = self.graph.region_slices.get("antenna")
        if antenna_slice is not None and antenna_slice.stop > antenna_slice.start:
            self._external_input[antenna_slice] += drive

    def _inject_cell_types(self, cell_types: tuple[str, ...], grid_flat: np.ndarray, *, gain: float):
        """Add ``gain * grid_flat[retinotopic_column]`` to each listed cell type."""
        if grid_flat.size == 0:
            return
        for name in cell_types:
            indices = self._cell_type_indices.get(name)
            if indices is None or indices.size == 0:
                continue
            if self.graph.column_of is None:
                self._external_input[indices] += gain * float(grid_flat.mean())
                continue
            cols = self.graph.column_of[indices]
            valid = cols >= 0
            if valid.any():
                selected = indices[valid]
                self._external_input[selected] += gain * grid_flat[cols[valid] % grid_flat.size]
            else:
                self._external_input[indices] += gain * float(grid_flat.mean())

    def step(
        self,
        frame_bgr: np.ndarray,
        sub_steps: int,
        audio_window: np.ndarray | None = None,
    ) -> StepStats:
        import time as _time

        t0 = _time.perf_counter()
        dt = self.dynamics._dt

        self._inject_vision(frame_bgr, dt)
        self._inject_hearing(audio_window)

        activity = None
        for _ in range(sub_steps):
            activity = self.dynamics.step(self._external_input)
            self._sim_time_s += dt

        stats = self._compute_stats(activity)
        stats.frame_process_ms = (_time.perf_counter() - t0) * 1000.0
        return stats

    def _compute_stats(self, activity: np.ndarray) -> StepStats:
        thresh = 0.5
        active_mask = activity > thresh
        region_activity = {}
        region_active_count = {}
        region_order = list(REGION_ORDER)
        if OTHER_REGION in self.graph.region_slices:
            region_order.append(OTHER_REGION)
        for r in region_order:
            sl = self.graph.region_slices.get(r)
            if sl is None:
                continue
            region_activity[r] = float(activity[sl].mean()) if sl.stop > sl.start else 0.0
            region_active_count[r] = int(active_mask[sl].sum())

        strongest, labels = self._strongest_active_connections(activity, top_k=5)

        audio = self._last_audio_features
        fly_state = compute_fly_state(
            region_activity,
            active_fraction=float(active_mask.mean()),
            motion_energy_mean=self._last_motion_energy,
            audio_level=float(audio.get("total", 0.0)),
            audio_onset=float(audio.get("onset", 0.0)),
        )
        emotion_mix = self.emotion_tracker.update(fly_state)

        return StepStats(
            sim_time_s=self._sim_time_s,
            active_neuron_count=int(active_mask.sum()),
            active_fraction=float(active_mask.mean()),
            region_activity=region_activity,
            region_active_count=region_active_count,
            mean_firing_rate_hz=float(self.dynamics.firing_rate_hz_estimate().mean()),
            strongest_connections=strongest,
            motion_energy_mean=self._last_motion_energy,
            frame_process_ms=0.0,
            strongest_connection_labels=labels,
            hearing_active=bool(self.config.hearing_enabled),
            audio_band_levels={
                name: float(audio.get(name, 0.0)) for name in ("low", "mid", "high")
            },
            audio_rms=float(audio.get("rms", 0.0)),
            audio_onset=float(audio.get("onset", 0.0)),
            fly_state=fly_state,
            emotion_mix=emotion_mix,
            emotion_mix_frames=self.emotion_tracker.sample_count,
        )

    def _strongest_active_connections(self, activity: np.ndarray, top_k: int = 5):
        """Strongest currently-active connections among the strongest synapses.

        Scanning all edges every frame is expensive for whole-brain graphs,
        so candidates are precomputed in ``__init__`` as the 60k largest by
        absolute weight.  For small synthetic graphs this is every edge.
        """
        if self._top_edge_src.size == 0:
            return [], []
        act_src = activity[self._top_edge_src]
        act_dst = activity[self._top_edge_dst]
        active = (act_src > 0.5) & (act_dst > 0.5)
        if not active.any():
            return [], []
        weighted = (
            np.abs(self._top_edge_w[active]) * act_src[active] * act_dst[active]
        )
        top = np.argsort(-weighted)[:top_k]
        src_all = self._top_edge_src[active]
        dst_all = self._top_edge_dst[active]
        w_all = self._top_edge_w[active]
        results = []
        labels = []
        for t in top:
            src = int(src_all[t])
            dst = int(dst_all[t])
            results.append((src, dst, float(w_all[t])))
            labels.append(
                f"{self.graph.label_for(src)} -> {self.graph.label_for(dst)}"
            )
        return results, labels
