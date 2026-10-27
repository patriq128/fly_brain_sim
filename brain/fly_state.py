"""
brain/fly_state.py
===================

An intentionally simple, clearly-labeled **projection** of the simulated
network activity into a few human-readable "internal state" axes, so the UI
can answer the playful question "what might the fly be feeling right now?".

WHAT THIS IS AND IS NOT
-----------------------
This module does **not** model affect, emotion, interoception, homeostasis,
or consciousness, and it makes no claim that Drosophila (or this
simulation) experiences anything. A real fly is not known to have subjective
feelings. What is computed here is a deterministic function of the same
region activity the simulation already produces: an illustrative UI summary,
on the same footing as the "firing rate (Hz)" display. Every axis below is
therefore **SIMULATION/ASSUMPTION**.

DERIVATION (all SIMULATION/ASSUMPTION)
--------------------------------------
Region activity values in [0, 1] are combined into:

    optic   = 0.30*retina + 0.25*lamina + 0.25*medulla + 0.20*lobula
    central = 0.50*central_complex + 0.50*mushroom_body
    motion  = min(1, motion_energy_mean / REF_MOTION_ENERGY)
    hearing = clip(audio_level, 0, 1)
    startle = clip(audio_onset, 0, 1)

    arousal    = clip(0.5*(active_fraction/REF_ACTIVE_FRACTION)
                      + 0.5*(mean_region_activity/REF_MEAN_ACTIVITY)
                      + 0.20*hearing, 0, 1)
    salience   = clip(0.50*optic + 0.40*motion + 0.25*hearing, 0, 1)
    escape     = clip(0.30*(lobula_plate/REF_LOBULA_PLATE)
                      + 0.60*motion + 0.25*startle, 0, 1)
    curiosity  = clip(central/REF_CENTRAL_ACTIVITY, 0, 1)
    valence    = clip(curiosity - escape, -1, 1)

The reference constants are deliberately chosen so that activity on the
bundled sample video spreads across the displayed bars; they are not
measured physiological thresholds. `escape` leans on the lobula plate
because that is where the real T4/T5 motion pathway terminates and where
optomotor/escape responses are read out; `curiosity` leans on the central
complex and mushroom body because those are the navigation and
learning/memory centres. `hearing`/`startle` use the Johnston's-organ
receptor model's band and onset outputs (see hearing/johnstons_organ.py);
the audible world is the video soundtrack, which is an ASSUMPTION. The
mapping from those real roles to a word like "curiosity" is the project's
invention.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Mapping

# Fixed display order and the labels `_classify` can return.  The emotion-mix
# graph always shows every label, including zero percentages, so the scale is
# stable across frames.
EMOTION_ORDER = (
    "Calm",
    "Settled",
    "Curious",
    "Alert",
    "Wary",
    "Startled",
    "Overwhelmed",
)

# SIMULATION/ASSUMPTION: the mix graph is the frequency of the hard mood
# labels over this many recent simulation frames (a rolling window), not a
# probability distribution or a model of graded emotion.
EMOTION_WINDOW_FRAMES = 120

# Reference scales for the normalized axes.  Arbitrary display calibration,
# not measured values (see module docstring).
REF_ACTIVE_FRACTION = 0.50
REF_MEAN_ACTIVITY = 0.80
REF_MOTION_ENERGY = 0.12
REF_LOBULA_PLATE = 0.60
REF_CENTRAL_ACTIVITY = 0.90


@dataclass(frozen=True)
class FlyState:
    """Illustrative activity-derived state shown in the UI (not an emotion)."""

    arousal: float       # 0..1 overall activation
    salience: float      # 0..1 how strongly vision is driving the brain
    escape: float        # 0..1 motion/lobula-plate "threat" drive
    curiosity: float     # 0..1 central-complex + mushroom-body activity
    valence: float       # -1..1 curiosity minus escape (approach/avoid axis)
    mood: str            # short UI label, one of a fixed set
    note: str            # one-line explanation of the label


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _region(region_activity: Mapping[str, float], key: str) -> float:
    return float(region_activity.get(key, 0.0))


def compute_fly_state(
    region_activity: Mapping[str, float],
    *,
    active_fraction: float,
    motion_energy_mean: float,
    audio_level: float = 0.0,
    audio_onset: float = 0.0,
) -> FlyState:
    """Project region activity into the illustrative state axes.

    Parameters are exactly the values already reported in ``StepStats``;
    this function has no state and no side effects.
    """
    retina = _region(region_activity, "retina")
    lamina = _region(region_activity, "lamina")
    medulla = _region(region_activity, "medulla")
    lobula = _region(region_activity, "lobula")
    lobula_plate = _region(region_activity, "lobula_plate")
    central_complex = _region(region_activity, "central_complex")
    mushroom_body = _region(region_activity, "mushroom_body")

    mean_activity = (
        sum(float(v) for v in region_activity.values()) / len(region_activity)
        if region_activity
        else 0.0
    )
    motion = min(1.0, max(0.0, float(motion_energy_mean) / REF_MOTION_ENERGY))
    hearing = _clip01(audio_level)
    startle = _clip01(audio_onset)
    optic = 0.30 * retina + 0.25 * lamina + 0.25 * medulla + 0.20 * lobula
    central = 0.50 * central_complex + 0.50 * mushroom_body

    arousal = _clip01(
        0.5 * (float(active_fraction) / REF_ACTIVE_FRACTION)
        + 0.5 * (mean_activity / REF_MEAN_ACTIVITY)
        + 0.20 * hearing
    )
    salience = _clip01(0.50 * optic + 0.40 * motion + 0.25 * hearing)
    escape = _clip01(
        0.30 * (lobula_plate / REF_LOBULA_PLATE)
        + 0.60 * motion
        + 0.25 * startle
    )
    curiosity = _clip01(central / REF_CENTRAL_ACTIVITY)
    valence = min(1.0, max(-1.0, curiosity - escape))

    mood, note = _classify(
        arousal=arousal,
        salience=salience,
        escape=escape,
        curiosity=curiosity,
        motion=motion,
    )
    return FlyState(
        arousal=arousal,
        salience=salience,
        escape=escape,
        curiosity=curiosity,
        valence=valence,
        mood=mood,
        note=note,
    )


def _classify(
    *,
    arousal: float,
    salience: float,
    escape: float,
    curiosity: float,
    motion: float,
) -> tuple[str, str]:
    """Map the axes to a fixed, documented set of UI labels."""
    if arousal < 0.25 and motion < 0.15:
        return "Calm", "little motion, low arousal"
    if escape >= 0.70 and motion >= 0.50:
        return "Startled", "strong optic-flow motion, high escape drive"
    if escape >= 0.55:
        return "Wary", "sustained motion in the visual field"
    if salience >= 0.55 and curiosity >= 0.55:
        return "Curious", "visual drive with an active central brain"
    if arousal >= 0.80:
        return "Overwhelmed", "very high activity across the brain"
    if salience >= 0.55:
        return "Alert", "strong visual drive, no strong escape response"
    if curiosity >= 0.70:
        return "Curious", "active central brain, little visual threat"
    return "Settled", "moderate activity, no strong threat"


class EmotionTracker:
    """Rolling frequency of the classified mood labels.

    This is deliberately just a histogram of the same hard labels shown as
    the current mood, over the last ``window`` simulation frames.  It is a
    transparent "how often has each label appeared recently" meter, not a
    graded-emotion model and not a claim of real experience.
    """

    def __init__(self, window: int = EMOTION_WINDOW_FRAMES):
        self.window = max(1, int(window))
        self._history: deque[str] = deque(maxlen=self.window)

    def reset(self) -> None:
        self._history.clear()

    @property
    def sample_count(self) -> int:
        """Number of frames currently in the rolling window."""
        return len(self._history)

    def update(self, state: FlyState) -> dict[str, float]:
        """Append one mood label and return per-label fractions in [0, 1]."""
        self._history.append(state.mood)
        counts = Counter(self._history)
        n = len(self._history)
        return {mood: counts.get(mood, 0) / n for mood in EMOTION_ORDER}
