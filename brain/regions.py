"""
brain/regions.py
=================

Defines the coarse brain regions used by this simulation and attaches a
short, honest "evidence label" to each one so the UI and docs can always
show the user what kind of claim is being made about that part of the model.

Evidence labels used throughout this project
---------------------------------------------
"BIOLOGY"     - An experimentally established anatomical/physiological fact
                (e.g. "the medulla is a layered neuropil downstream of the
                lamina", "T4/T5 cells are direction-selective").
"CONNECTOME"  - Derived from (or, in DEMO/DETAILED mode, *inspired by
                published summaries of*) real connectome measurements
                (neuron/cell-type counts, coarse connectivity motifs).
"SIMULATION"  - A mathematical/numerical modeling choice (e.g. leaky-rate
                neuron dynamics, sigmoid activation, Euler integration).
"ASSUMPTION"  - A simplification or design decision made for this project
                that has no direct biological claim (e.g. exact 3D layout
                coordinates used only for visualization, or feeding a
                video file - a fly does not watch video - into the retina
                model).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Evidence(str, Enum):
    BIOLOGY = "biology"
    CONNECTOME = "connectome"
    SIMULATION = "simulation"
    ASSUMPTION = "assumption"


@dataclass(frozen=True)
class RegionInfo:
    key: str
    display_name: str
    description: str
    evidence: Evidence
    # Coarse 3D anchor position for visualization only, in a unit-ish cube.
    # ASSUMPTION: this is a schematic layout that preserves left/right and
    # anterior/posterior *relationships* described qualitatively in the
    # literature (optic lobes lateral, central complex central, mushroom
    # bodies dorsal), NOT real morphological coordinates from any atlas.
    anchor_left: tuple[float, float, float]
    anchor_right: tuple[float, float, float]
    color: str


# Order matters: it is roughly the feedforward flow of visual information,
# which IS a real, well-documented anatomical pathway:
#   retina -> lamina -> medulla -> lobula / lobula plate -> central brain
# (see e.g. Fischbach & Dittrich 1989 for the classical optic lobe anatomy,
# and the flyvis / connectome papers cited in README.md for the modern
# connectomic description of this pathway).
REGIONS: dict[str, RegionInfo] = {
    "retina": RegionInfo(
        key="retina",
        display_name="Retina (photoreceptors)",
        description=(
            "Ommatidial photoreceptors (R1-R8). BIOLOGY: the compound eye "
            "samples the visual world through ~750-800 ommatidia per eye, "
            "each containing 8 photoreceptor cells with different spectral "
            "and functional roles (R1-R6 outer, achromatic, motion-relevant; "
            "R7-R8 inner, color-relevant)."
        ),
        evidence=Evidence.BIOLOGY,
        anchor_left=(-3.2, 0.0, 0.3),
        anchor_right=(3.2, 0.0, 0.3),
        color="#f2c14e",
    ),
    "lamina": RegionInfo(
        key="lamina",
        display_name="Lamina",
        description=(
            "First optic ganglion. BIOLOGY: lamina monopolar cells (LMCs, "
            "e.g. L1-L5) perform temporal filtering and contrast "
            "enhancement on photoreceptor output; L1/L2 famously split "
            "signals into ON and OFF channels."
        ),
        evidence=Evidence.BIOLOGY,
        anchor_left=(-2.6, 0.0, 0.15),
        anchor_right=(2.6, 0.0, 0.15),
        color="#f78154",
    ),
    "medulla": RegionInfo(
        key="medulla",
        display_name="Medulla",
        description=(
            "Second, largest optic ganglion. BIOLOGY: contains dozens of "
            "columnar cell types (Mi, Tm, T4, etc.) organized retinotopically "
            "in ~10 layers; this is where local motion detection begins "
            "(T4 dendrites here)."
        ),
        evidence=Evidence.BIOLOGY,
        anchor_left=(-2.0, 0.0, 0.0),
        anchor_right=(2.0, 0.0, 0.0),
        color="#4d9078",
    ),
    "lobula": RegionInfo(
        key="lobula",
        display_name="Lobula",
        description=(
            "Third optic ganglion. BIOLOGY: integrates medulla output; "
            "involved in looming/figure-ground and object-motion "
            "sensitive processing (e.g. LC (lobula columnar) neurons)."
        ),
        evidence=Evidence.BIOLOGY,
        anchor_left=(-1.5, 0.3, -0.1),
        anchor_right=(1.5, 0.3, -0.1),
        color="#5b85aa",
    ),
    "lobula_plate": RegionInfo(
        key="lobula_plate",
        display_name="Lobula plate",
        description=(
            "Fourth optic ganglion, adjacent to the lobula. BIOLOGY: "
            "T4 (ON) and T5 (OFF) direction-selective cells terminate here "
            "in four layers tuned to the four cardinal motion directions, "
            "onto large 'wide-field' tangential cells (e.g. HS, VS) that "
            "encode optic flow for course control."
        ),
        evidence=Evidence.BIOLOGY,
        anchor_left=(-1.4, -0.4, -0.1),
        anchor_right=(1.4, -0.4, -0.1),
        color="#8860d0",
    ),
    "central_complex": RegionInfo(
        key="central_complex",
        display_name="Central complex",
        description=(
            "BIOLOGY: a central, midline brain structure (ellipsoid body, "
            "fan-shaped body, protocerebral bridge) implicated in spatial "
            "orientation, heading representation and navigation, "
            "integrating visual and self-motion signals."
        ),
        evidence=Evidence.BIOLOGY,
        anchor_left=(-0.3, 0.1, 0.1),
        anchor_right=(0.3, 0.1, 0.1),
        color="#d64550",
    ),
    "mushroom_body": RegionInfo(
        key="mushroom_body",
        display_name="Mushroom body",
        description=(
            "BIOLOGY: paired structures built from ~2,000 Kenyon cells "
            "per hemisphere; the fly's principal learning/memory center, "
            "associating sensory patterns (including visual features) "
            "with valence."
        ),
        evidence=Evidence.BIOLOGY,
        anchor_left=(-0.9, 0.5, 0.6),
        anchor_right=(0.9, 0.5, 0.6),
        color="#3d348b",
    ),
    # Only present in real-connectome (FlyWire) mode: neurons whose published
    # cell class is outside the seven coarse regions above (olfactory,
    # mechanosensory, ascending/descending, etc.).  Kept so the whole-brain
    # graph is complete rather than silently dropping a third of the brain.
    "other": RegionInfo(
        key="other",
        display_name="Other central brain",
        description=(
            "CONNECTOME: real FlyWire neurons not assigned to one of the "
            "seven visual/central regions this project models (olfactory, "
            "mechanosensory, ascending/descending, unclassified). They are "
            "included in whole-brain mode but receive no targeted visual "
            "drive."
        ),
        evidence=Evidence.CONNECTOME,
        anchor_left=(0.0, -0.4, -0.3),
        anchor_right=(0.0, -0.4, -0.3),
        color="#7a7a7a",
    ),
}

# ASSUMPTION: This project only implements one downstream "decision" style
# readout (a simple pooled central-complex activity summary; see
# brain/network.py). It does NOT claim this corresponds to any specific
# fly behavior (e.g. turning, landing, escape) - it is presented purely as
# an illustrative aggregate signal.
REGION_ORDER = [
    "retina",
    "lamina",
    "medulla",
    "lobula",
    "lobula_plate",
    "central_complex",
    "mushroom_body",
]

# Region key used for real-connectome neurons outside REGION_ORDER.  It is
# deliberately not part of REGION_ORDER: synthetic graphs have no such bucket.
OTHER_REGION = "other"
