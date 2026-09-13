# Simulation Notes: What Is Biology, What Is Not

This document is the single place that spells out, stage by stage, what in
this project is an experimentally established fact, what is derived from a
real connectome, what is a generic mathematical/simulation choice, and what
is a simplification or assumption made purely so the project runs. Every
module docstring in the codebase makes the same distinction locally; this
file collects it in one place for reference.

**This project is a computational simulation *inspired by* the Drosophila
visual system and connectome. It does not reproduce, and does not claim to
reproduce, the actual thoughts, percepts, or consciousness of a real fly.**

## 1. Evidence categories used throughout the code

| Label | Meaning |
|---|---|
| `BIOLOGY` | An experimentally established anatomical or physiological fact from the published literature. |
| `CONNECTOME` | Taken from, or (in DEMO/DETAILED mode) inspired by published summaries of, real measured connectivity. |
| `SIMULATION` | A mathematical/numerical modeling choice (activation function, integration scheme, etc.) with no direct biological claim. |
| `ASSUMPTION` | A simplification or design decision specific to this project (e.g. using video as visual input, schematic 3D coordinates). |

## 2. Stage-by-stage breakdown

### Retina / photoreceptors (`vision/retina.py`)
- **BIOLOGY**: the compound eye has ~750-800 ommatidia per eye in a
  hexagonal lattice, each with 8 photoreceptors (R1-R8); R1-R6 are
  achromatic and feed the motion pathway via the lamina, R7-R8 are
  color/UV-relevant and project more directly into the medulla (Hardie
  1985; Ready, Hanson & Benzer 1976). Fly photoreceptors have unusually
  high temporal bandwidth, i.e. they are built for detecting fast changes,
  not sharp static images (Juusola & Hardie 2001).
- **ASSUMPTION**: we sample an ordinary RGB video frame instead of a real
  fly's visual world. Real fly photoreceptors are UV/blue/green sensitive
  and not red-sensitive the way human cones are (Salcedo et al. 1999); we
  do not attempt a physically accurate spectral remapping, we just use
  luminance for the R1-R6-like channel and a coarse blue/green-weighted
  proxy for the R7-R8-like channel.
- **SIMULATION**: the sampling grid resolution (16x16 for DEMO, 32x48 for
  DETAILED) is a computational choice, not a measured acuity limit. Light
  adaptation is modeled as an exponential moving average baseline
  subtraction - a generic gain-control idea, not Juusola & Hardie's
  biophysical adaptation model.

### Lamina / early motion pathway (`brain/motion_pathway.py`)
- **BIOLOGY**: T4 (ON) and T5 (OFF) cells are the fly's identified
  direction-selective neurons, each with four subtypes tuned to the four
  cardinal directions, and their required input structure (one delayed,
  one non-delayed input per direction) matches the classical
  Hassenstein-Reichardt correlator (HRC) model (Borst & Egelhaaf 1989),
  now confirmed at synaptic resolution by connectome work (Shinomiya et
  al. 2019) and reproduced from connectivity + optimization by
  Lappalainen et al. 2024 (*Nature*, the flyvis model,
  https://github.com/TuragaLab/flyvis).
- **SIMULATION**: `ReichardtMotionDetector` implements a textbook 2-input
  HRC with a single exponential delay filter, vectorized over the whole
  grid with `numpy.roll` for the four cardinal directions. Real T4/T5
  dendrites integrate ~4+ distinct upstream cell types with specific
  nonlinear synapses; this is a deliberate reduction to the minimal
  correlator model, not a reproduction of the full circuit.

### Connectome / wiring (`brain/connectome.py`, `data/connectome_loader.py`)
- **CONNECTOME (real, when used)**: `--level connectome` loads a real
  edge list you export yourself from neuPrint (hemibrain) or FlyWire (see
  `data/connectome_loader.py` for exact instructions and CSV schema). When
  used, neuron counts, connectivity, and (if you supply it) region labels
  come directly from that data.
- **CONNECTOME-inspired, not real (DEMO/DETAILED default)**: neuron
  counts per region are chosen to be proportionally plausible (the optic
  lobes dominate total neuron count, matching the qualitative fact that a
  large fraction of the fly brain is visual) and the region ordering
  follows the real feedforward optic-lobe pathway. The *specific* neuron
  counts, fan-in/fan-out numbers, and connection weights are procedurally
  generated and invented for this project - they are not read from FlyWire
  or the hemibrain. `mushroom_body` in DETAILED mode uses ~2,000 neurons
  per hemisphere because that is the approximate real order of magnitude
  of Kenyon cell counts; that specific number is the one deliberately
  chosen "real-ish" data point among otherwise synthetic counts.
- **SIMULATION**: lateral inhibition, convergent pooling ratios, and the
  weight-normalization step (`_normalize_incoming_weights`) that keeps
  the network from saturating are numerical-stability choices, not
  measurements. Real synaptic scaling/homeostasis exists (Turrigiano
  2008) but the rule used here is not modeling that phenomenon
  quantitatively - it is just there so the demo doesn't saturate to 100%
  activity immediately.

### Neuron dynamics (`brain/neuron_model.py`)
- **SIMULATION**: a standard leaky-rate neuron, `tau*dV/dt = -V + input`,
  with a sigmoid activation. This produces smooth, interpretable
  "activity" in [0, 1] suitable for real-time visualization. It is NOT a
  spiking model and is not fit to any Drosophila electrophysiology
  recording. The "firing rate (Hz)" shown in the stats panel is an
  illustrative linear rescaling of this activity value into 0-200 Hz, not
  a measured or calibrated rate - this is stated directly in the on-screen
  stats text.

### Higher brain regions: lobula, lobula plate, central complex, mushroom body
- **BIOLOGY**: these are real, named neuropils with well-documented broad
  roles - lobula plate houses the real HS/VS wide-field tangential cells
  used for optic-flow-based course control; the central complex
  (ellipsoid body, fan-shaped body, protocerebral bridge) is implicated in
  heading/spatial navigation; the mushroom body is the fly's
  learning/memory center built from ~2,000 Kenyon cells per hemisphere.
- **ASSUMPTION**: this project does NOT implement any specific,
  identified circuit for these regions (no real HS/VS cell model, no real
  Kenyon cell sparse-coding model, no real central-complex ring-attractor
  model). They are populated with generic leaky-rate neurons that receive
  pooled input from upstream regions and recurrent connections among
  themselves, purely to give the 3D visualization something coherent and
  anatomically-labeled to show downstream of the visual pathway. No claim
  is made that their activity in this simulation corresponds to any real
  fly behavior, memory, or navigational state.

### 3D layout (`brain/regions.py`, `visualization/brain3d.py`)
- **ASSUMPTION**: neuron 3D coordinates are a schematic layout (optic
  lobes lateral, central complex central, mushroom bodies dorsal) chosen
  only so the visualization reads as a recognizable, labeled brain
  diagram. They are NOT real morphological coordinates from any atlas or
  skeleton reconstruction (those are available from neuPrint/FlyWire but
  are not used here).

## 3. Simulation levels and their computational cost

| Level | Neurons | Synapses (approx) | Source | Typical use |
|---|---|---|---|---|
| `demo` | ~1,400 | ~12,000 | fully synthetic | interactive on a laptop CPU, ~300+ simulation ticks/sec measured headless in this repo's test environment |
| `detailed` | ~15,000 | proportionally more (still capped by fan-in/fan-out limits in `brain/connectome.py`) | fully synthetic | still interactive; scipy sparse matrix-vector products keep the per-step cost roughly linear in edge count |
| `connectome` | up to ~139,000 (full FlyWire) or ~25,000 (hemibrain) | up to ~16.5M (FlyWire) or ~20M (hemibrain) | real, user-supplied | **not recommended to load whole-brain, dense, real data for real-time interactive use** - see below |

**Why we do not attempt real-time simulation of the full connectome:**
a dense recurrent step over the full FlyWire graph is a sparse
matrix-vector product over ~16.5 million weighted edges, once per neural
integration sub-step, on top of maintaining ~139k neuron state vectors and
their history for visualization (points, colors, edge subsampling, etc.)
every rendered frame. This is not impossible on capable hardware, but it
is well outside what this project's default settings target ("runs on a
normal computer", per the project spec). Recommended paths if you want to
push toward `connectome` scale:

1. **Restrict to a sub-circuit.** Use neuprint-python / FlyWire's
   annotations to pull only the optic lobe + a handful of downstream ROIs
   (as the example in `data/connectome_loader.py` does) rather than the
   whole brain. This gets you real connectivity at a scale (thousands to
   tens of thousands of neurons) close to this project's `detailed` mode.
2. **Reduce the edges drawn in 3D, not the edges simulated.** The
   simulation's sparse dynamics can handle far more edges than the 3D
   viewer should ever try to draw as individual lines;
   `visualization/brain3d.py` already subsamples to `MAX_EDGES_DRAWN`
   edges for rendering while `brain/neuron_model.py` simulates the full
   loaded graph - raise `MAX_EDGES_DRAWN` cautiously.
3. **Lower `--sub-steps` and `--video-fps`.** Neural integration cost
   scales with `steps_per_video_frame`; reducing this trades dynamical
   smoothness for speed.
4. **Move to GPU.** `brain/neuron_model.py` only depends on `scipy.sparse`
   matrix-vector products and elementwise numpy ops - both have direct
   GPU-accelerated equivalents (e.g. `cupyx.scipy.sparse`, or reimplementing
   `LeakyRateNetwork` in PyTorch/JAX using its own sparse or dense
   representations, as flyvis does for the real visual-system-only
   connectome). This is the most promising path to true full-connectome,
   real-time simulation and is the top suggestion in `README.md`'s future
   work section.

## 4. Sources cited in this project

- Dorkenwald, S. et al. (2024). *Neuronal wiring diagram of an adult brain.*
  Nature. https://www.nature.com/articles/s41586-024-07558-y (FlyWire/FAFB
  whole-brain connectome, ~139,000 neurons)
- Schlegel, P. et al. (2024). *Whole-brain annotation and multi-connectome
  cell typing of Drosophila.* Nature.
  https://www.nature.com/articles/s41586-024-07686-5
- Scheffer, L.K. et al. (2020). *A connectome and analysis of the adult
  Drosophila central brain.* eLife. (the "hemibrain" dataset,
  https://www.janelia.org/project-team/flyem/hemibrain)
- Lappalainen, J.K. et al. (2024). *Connectome-constrained networks predict
  neural activity across the fly visual system.* Nature.
  https://doi.org/10.1038/s41586-024-07939-3
  (flyvis, https://github.com/TuragaLab/flyvis - the direct inspiration for
  this project's "simulate visual processing from connectome structure,
  don't just train an arbitrary network" approach)
- Borst, A. & Egelhaaf, M. (1989). *Principles of visual motion detection.*
  Trends in Neurosciences. (classical Hassenstein-Reichardt correlator
  theory underlying `brain/motion_pathway.py`)
- Shinomiya, K. et al. (2019). *Comparisons between the ON- and OFF-edge
  motion pathways in the Drosophila brain.* eLife. (connectomic
  confirmation of T4/T5 input structure)
- Hardie, R.C. (1985) and Ready, Hanson & Benzer (1976) - classical
  Drosophila photoreceptor anatomy/physiology referenced in
  `vision/retina.py`.
- Salcedo, E. et al. (1999). *Blue- and green-absorbing visual pigments of
  Drosophila.* J. Neuroscience. (photoreceptor spectral sensitivity)
- Turrigiano, G. (2008). *The self-tuning neuron: synaptic scaling of
  excitatory synapses.* Cell. (referenced re: the weight-normalization
  analogy, not implemented quantitatively)
- Chiappe, M.E. et al. and later central-complex heading literature -
  general background for the central complex's role in navigation; no
  specific ring-attractor model is implemented here.
- Ramdya lab, EPFL: NeuroMechFly / FlyGym
  (https://pypi.org/project/flygym) - example of a full embodied fly
  simulation (biomechanics + sensorimotor control) that this project does
  NOT attempt; cited in README.md as related work and future-direction
  inspiration (coupling this visual/brain model to a body).
