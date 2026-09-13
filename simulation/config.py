"""
simulation/config.py
=====================

Defines the three simulation "levels" described in the project README:

    DEMO        - small synthetic network, runs interactively on a laptop CPU.
    DETAILED    - larger synthetic network with more columns / cell types,
                  still procedurally generated but denser and closer in
                  proportion to published cell-type counts.
    CONNECTOME  - structured to accept a real connectivity matrix exported
                  from a connectome database (FlyWire / hemibrain via
                  neuprint-python or the FlyWire "codex" CSV export). This
                  repository does NOT bundle real connectome data (it is
                  tens of millions of rows and access is governed by the
                  FlyWire/Janelia data licenses) - `data/connectome_loader.py`
                  documents exactly how to plug it in.

IMPORTANT - HONESTY ABOUT WHAT IS AND ISN'T "REAL" DATA
--------------------------------------------------------
Only CONNECTOME mode, when pointed at a real exported adjacency file, uses
actual measured connectivity. DEMO and DETAILED modes use a *procedurally
generated* graph whose region sizes, layer ordering and broad connection
motifs are *inspired by* published descriptions of the fly optic lobe and
central brain (retinotopic feedforward pathway, lateral inhibition, few
long-range "wide-field" neurons) but the specific neuron count, weights and
wiring are NOT taken from any dataset. This is stated in the UI and README
as well - see docs/SIMULATION_NOTES.md for the full biology/assumption
breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SimLevel(str, Enum):
    DEMO = "demo"
    DETAILED = "detailed"
    CONNECTOME = "connectome"


@dataclass
class RegionSizes:
    """Neuron counts per coarse brain region for a given simulation level.

    Region names and their broad roles are real neuroanatomical terms used
    throughout the Drosophila literature (retina/photoreceptors, lamina,
    medulla, lobula, lobula plate, central complex, mushroom body). The
    *counts* below are NOT real cell-type counts except where noted; they
    are chosen so the demo/detailed graphs are runnable on a laptop while
    remaining proportionally sensible (e.g. the optic lobes dominate neuron
    count, matching the qualitative fact that a large fraction of the fly
    brain is devoted to vision).
    """

    retina: int
    lamina: int
    medulla: int
    lobula: int
    lobula_plate: int
    central_complex: int
    mushroom_body: int

    @property
    def total(self) -> int:
        return (
            self.retina
            + self.lamina
            + self.medulla
            + self.lobula
            + self.lobula_plate
            + self.central_complex
            + self.mushroom_body
        )


# Ommatidia counts: a real adult Drosophila compound eye has on the order of
# ~750-800 ommatidia per eye (Ready, Hanson & Benzer 1976, and many later
# papers). DEMO deliberately undersamples this for speed; DETAILED gets much
# closer to the real number. CONNECTOME mode's retina size is instead
# determined by whatever real data is loaded.
REGION_SIZES = {
    SimLevel.DEMO: RegionSizes(
        retina=256,          # undersampled hex-ish grid (~1/3 of one real eye)
        lamina=256,          # ~1:1 with retina columns (LMCs are columnar)
        medulla=512,         # medulla has many more cell types per column
        lobula=128,
        lobula_plate=64,     # home of the real, well-studied HS/VS wide-field cells
        central_complex=64,
        mushroom_body=96,
    ),
    SimLevel.DETAILED: RegionSizes(
        retina=1536,         # ~2 eyes' worth of ommatidia (both eyes combined)
        lamina=1536,
        medulla=4096,
        lobula=1024,
        lobula_plate=256,
        central_complex=512,
        mushroom_body=2000,  # order of magnitude of real Kenyon cell counts (~2000/hemisphere)
    ),
    # Placeholder only - real CONNECTOME mode replaces this with counts read
    # from the loaded adjacency file (see data/connectome_loader.py).
    SimLevel.CONNECTOME: RegionSizes(
        retina=1536,
        lamina=1536,
        medulla=4096,
        lobula=1024,
        lobula_plate=256,
        central_complex=512,
        mushroom_body=2000,
    ),
}

# Approximate, PUBLISHED whole-brain / hemibrain scale, for reference and for
# the README's "what would it take to go further" discussion. These numbers
# are cited in docs/SIMULATION_NOTES.md and are not used directly as
# simulation parameters because simulating them densely and interactively is
# not practical on consumer hardware (see that doc for the reasoning).
PUBLISHED_SCALE_REFERENCE = {
    "hemibrain_neurons": 25000,       # Scheffer et al. 2020, eLife (hemibrain)
    "hemibrain_synapses": 20_000_000,
    "flywire_fafb_neurons": 139255,   # Dorkenwald et al. 2024 / Schlegel et al. 2024, Nature
    "flywire_fafb_synapses": 16_500_000,  # approx. weighted edges, Nature 2024 papers
}


@dataclass
class NeuronDynamicsParams:
    """Parameters for the generic leaky-rate neuron model (brain/neuron_model.py).

    These are a standard, generic leaky-integrator parameterization (the
    same family of equations used across countless rate-based neural
    simulations) and are NOT fitted to or taken from Drosophila
    electrophysiology. They are simplifications, chosen for numerical
    stability and visually interpretable dynamics at video frame-rate
    timescales.
    """

    tau_ms: float = 20.0            # membrane-like time constant
    dt_ms: float = 5.0              # integration step
    gain: float = 1.0               # input gain into the sigmoid activation
    activation_slope: float = 4.0   # sigmoid steepness
    activation_threshold: float = 0.15
    noise_std: float = 0.01         # small stochastic jitter for visual liveliness
    refractory_decay: float = 0.85  # simple adaptation/fatigue factor


@dataclass
class SimulationConfig:
    level: SimLevel = SimLevel.DEMO
    region_sizes: RegionSizes = field(default_factory=lambda: REGION_SIZES[SimLevel.DEMO])
    dynamics: NeuronDynamicsParams = field(default_factory=NeuronDynamicsParams)
    retina_grid_shape: tuple[int, int] = (16, 16)   # rows x cols hex-ish sampling grid
    connectome_path: str | None = None              # required for CONNECTOME level
    video_target_fps: float = 15.0                  # frames/sec fed to the sim
    steps_per_video_frame: int = 3                  # brain integration sub-steps per frame
    random_seed: int = 0

    @staticmethod
    def for_level(level: SimLevel, connectome_path: str | None = None) -> "SimulationConfig":
        sizes = REGION_SIZES[level]
        grid = {
            SimLevel.DEMO: (16, 16),
            SimLevel.DETAILED: (32, 48),
            SimLevel.CONNECTOME: (32, 48),
        }[level]
        return SimulationConfig(
            level=level,
            region_sizes=sizes,
            retina_grid_shape=grid,
            connectome_path=connectome_path,
        )
