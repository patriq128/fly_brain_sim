# Simulation Notes: What Is Biology, What Is Not

This document is the single place that spells out, stage by stage, what in
this project is an experimentally established fact, what is derived from a
real connectome, what is a generic mathematical/simulation choice, and what
is a simplification or assumption made purely so the project runs. Every
module docstring in the codebase makes the same distinction locally; this
file collects it in one place for reference.

**This project is a computational simulation *inspired by* the Drosophila
visual and auditory systems and connectome. It does not reproduce, and does
not claim to reproduce, the actual thoughts, percepts, or consciousness of a
real fly.**

## 1. Evidence categories used throughout the code

| Label | Meaning |
|---|---|
| `BIOLOGY` | An experimentally established anatomical or physiological fact from the published literature. |
| `CONNECTOME` | Taken from, or (in DEMO/DETAILED mode) inspired by published summaries of, real measured connectivity. |
| `SIMULATION` | A mathematical/numerical modeling choice (activation function, integration scheme, etc.) with no direct biological claim. |
| `ASSUMPTION` | A simplification or design decision specific to this project (e.g. using video/soundtrack as sensory input, schematic 3D coordinates). |

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

### Auditory pathway / Johnston's organ (`hearing/`)
- **BIOLOGY**: Johnston's organ is a chordotonal organ in the antenna and
  the fly's main auditory organ. Its mechanosensory JO neurons (cell types
  `JO-A`…`JO-F` plus many subtypes) are the auditory receptor neurons and
  project to the antennal mechanosensory and motor center (AMMC), which
  relays sound/vibration to the central brain. Drosophila hearing is tuned
  to near-field sound and vibration, including the ~200 Hz pulse component
  of the male courtship song; related JO/AMMC pathways also carry wind and
  gravity information (Kamikouchi et al. 2006, 2009; Albert & Göpfert
  2015). JO neurons adapt to sustained sound and are especially sensitive
  to transients.
- **ASSUMPTION**: the audio track of the input video is treated as the
  fly's acoustic world, exactly as video frames are treated as its visual
  world. We do not model near-field acoustics, antennal mechanics, or the
  measured frequency tuning of individual JO subtypes. The audio is
  analysed but not played aloud by the app.
- **SIMULATION**: `hearing/johnstons_organ.py` computes three FFT-band
  amplitudes (low ~80–180 Hz, mid ~180–350 Hz, high ~350–700 Hz) from a
  Hann-windowed short-time window, applies an exponential moving-average
  adaptation baseline per band, and reports a rectified frame-to-frame
  broadband onset signal. The three bands plus onset are combined with
  fixed weights (mid band weighted most, since it contains the courtship
  pulse frequency) and injected as external current into auditory receptor
  targets. In `connectome` mode the targets are the real `JO-*` neurons in
  the loaded graph (mapped to the `antenna` region); in `demo`/`detailed`
  mode they are a small synthetic `antenna` population. If neither exists
  in a custom CSV export, the `other` bucket is used as a fallback.
- **CONNECTOME**: the `JO-*` cell-type labels and the neurons' measured
  synapses are from the Schlegel et al. 2024 annotations/FlyWire graph. The
  default ~20k subset contains ~240 JO neurons; whole-brain mode contains
  all ~1,100. The synthetic `antenna` count and its antenna → central
  complex / mushroom body wiring are invented for this project.
- **ASSUMPTION**: all `JO-*` neurons receive the same band mixture and
  injection gain; `--audio-gain` scales the drive and `--no-hearing`
  disables it. The band ranges, gains and adaptation constants are
  display/behavior calibration choices, not fitted electrophysiology.

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

### Connectome / wiring (`brain/connectome.py`, `data/flywire_source.py`, `data/connectome_loader.py`)
- **CONNECTOME (real, default FlyWire path)**: `--level connectome` with
  `--connectome-source flywire` (the default when no `--connectome-path` is
  given) downloads and verifies the public FlyWire FAFB v783 proofread
  connectivity table (852 MB, ~16.85 M `(pre, post, neuropil)` synapse-count
  rows; Zenodo `10.5281/zenodo.10676866`) and the Schlegel et al. 2024
  annotation table (27 MB, 139,255 neurons; Nature supplementary data). The
  following are **measured data**: neuron root ids, cell types, soma
  coordinates, synapse counts per directed pair, and predicted/known
  neurotransmitter. The default interactive graph is a ~20,000-neuron subset;
  `--max-neurons 0` builds the whole 139,255-neuron brain (~2.70 M weighted
  edges at the published ≥5-synapse threshold). Source files are cached under
  the git-ignored `data/real/` directory.
- **CONNECTOME (real, CSV path)**: `--connectome-path` loads a
  `neurons.csv`/`edges.csv` export you supply (e.g. hemibrain via
  neuprint-python), documented in `data/connectome_loader.py`. Neuron ids,
  edges, optional regions/cell types/transmitters are then measured from
  that export.
- **SIMULATION/ASSUMPTION in the real-connectome path** (all called out in
  `data/flywire_source.py`):
  - The ~180+ published cell classes are collapsed into eight coarse regions
    plus an `other` bucket; T4/T5 are assigned to `lobula_plate` because that
    is where their axons terminate and the motion readout lives, and
    Johnston's-organ `JO-*` neurons are assigned to `antenna`.
  - Per-neuron transmitter is mapped to an effective sign (ACh +1; GABA,
    glutamate, histamine, glycine −1; dopamine/serotonin/octopamine +0.25).
    Biologically, photoreceptors release inhibitory histamine; here their
    output is **forced excitatory** because the retina model emits a
    contrast/luminance drive rather than a histamine release rate. Without
    that override a bright frame would silence the pathway in this rate-model
    abstraction.
  - Synapse counts are compressed with `log1p`, and each neuron's total
    absolute incoming weight is capped (`DEFAULT_WEIGHT_TARGET = 1.5`) so a
    few huge fan-in neurons do not saturate the generic rate model.
  - The ~20k subset is chosen by whole retinotopic tiles rather than random
    neurons, to preserve local columnar circuits. This is a display/compute
    budget choice, not a biological claim.
  - External visual drive is injected only into identified photoreceptors
    (R1-6 get the achromatic channel, R7/R8 the chroma proxy), plus
    optionally the four T4/T5 subtype maps from the HRC. Auditory drive is
    injected only into identified `JO-*` Johnston's-organ neurons (see the
    auditory-pathway section above). Every other region is driven solely
    through the measured synapses. **ASSUMPTION**: the
    video's +x axis is identified with T4/T5 subtype `a` (front-to-back in
    the real fly; Maisak et al. 2013) and its four cardinal directions are
    assigned as a/b/c/d = right/left/up/down. The absolute azimuth is
    arbitrary for a camera video. `--no-motion-injection` disables the
    T4/T5 drive and leaves the connectome dynamics to propagate the
    photoreceptor signal alone.
  - Soma coordinates are converted from FAFB voxels to a physically
    proportioned scene and centered; optic-lobe soma position in the
    (anterior-posterior, dorsal-ventral) plane is used as a coarse proxy for
    visual-field position (the retinotopic column map). The fly optic lobe is
    genuinely retinotopic, but this particular mapping is not a measured
    retinotopic atlas.
  - The leaky-rate dynamics are the same generic model as every other level.
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

### Real-data provenance and licensing (`data/real/`)
- The FlyWire connectivity table and Schlegel et al. annotations are
  **CC-BY-4.0**. They are downloaded on first use and verified against the
  published size and checksum (MD5/SHA-256); `PROVENANCE.json` records the
  exact URL, DOI, size, checksum and fetch time. Nothing large is committed
  to this repository.
- If you publish results derived from this data, cite Dorkenwald et al. 2024
  and Schlegel et al. 2024 (also Eckstein, Bates et al. 2024 for the
  neurotransmitter predictions). See `data/real/README.md` and §4 below.

### Neuron dynamics (`brain/neuron_model.py`)
- **SIMULATION**: a standard leaky-rate neuron, `tau*dV/dt = -V + input`,
  with a sigmoid activation. This produces smooth, interpretable
  "activity" in [0, 1] suitable for real-time visualization. It is NOT a
  spiking model and is not fit to any Drosophila electrophysiology
  recording. The "firing rate (Hz)" shown in the stats panel is an
  illustrative linear rescaling of this activity value into 0-200 Hz, not
  a measured or calibrated rate - this is stated directly in the on-screen
  stats text.

### Simulated "feeling" readout (`brain/fly_state.py`)
- **SIMULATION/ASSUMPTION**: the stats overlay grows a playful
  "what is the fly feeling?" block from the same region activity. It
  computes five fixed axes:
  - `arousal`: weighted overall active fraction + mean region activity +
    a small hearing-level term.
  - `salience`: optic-lobe activity, motion energy, and hearing level.
  - `escape` ("fear"): lobula-plate activity, motion energy, and the
    auditory onset/transient signal. The lobula plate is where the real
    T4/T5 motion pathway terminates and optomotor/escape responses are read
    out; JO-receptor transients are an additional startle-like input.
  - `curiosity`: central-complex + mushroom-body activity. Those are the
    real navigation and learning/memory centres, but mapping their generic
    simulated activity to a word like "curiosity" is the project's
    invention.
  - `valence`: `curiosity - escape`, an approach/avoid-style axis.
  A fixed rule set turns the axes into one of `Calm`, `Settled`,
  `Curious`, `Alert`, `Wary`, `Startled`, or `Overwhelmed`. The reference
  scales and thresholds are display calibration, not measured physiology.
- **Emotion-mix graph**: the percentages shown in the stats overlay and in
  the video panel's colored bar chart are a rolling histogram of those hard
  labels over the last `EMOTION_WINDOW_FRAMES = 120` simulation frames. They
  answer "what fraction of recent frames was classified as each label", not
  "how strongly does the fly feel each emotion". The window is cleared on
  restart. Highlighting the current label and the bar colors are display
  choices.
- **NOT CLAIMED**: a real fly is not known to have subjective feelings, and
  this module does not model affect, emotion, interoception, homeostasis or
  consciousness. It is a deterministic UI summary of simulated activity,
  in the same category as the illustrative "firing rate (Hz)" number. The
  overlay states this explicitly on screen ("ILLUSTRATIVE - not a real
  experience").

### End-of-video behavior summary (`simulation/behavior_log.py`, `visualization/behavior_summary.py`)
- **SIMULATION/ASSUMPTION**: the app records one lightweight snapshot per
  displayed video frame (active fraction, per-region mean activity, the
  illustrative fly-state axes, motion energy, and auditory band levels).
  When the clip first reaches its end, it writes a summary PNG containing a
  time-aligned mood strip, a region-activity heatmap, the state axes over
  time, and the fraction of frames spent in each mood, then pauses and
  displays it.
- The summary is **descriptive of this simulation only**. It adds no new
  modeling and is not a fly ethogram, behavioral classification, or claim
  that the activity corresponds to anything a real fly does. The mood labels
  are the same playful UI projection described above; the region heatmap is
  the same generic leaky-rate activity shown live in the app. The chart is
  saved under `data/summary/` (git-ignored) and can be disabled with
  `--no-summary-graph`.

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
| `demo` | ~1,440 | ~12,700 | fully synthetic | interactive on a laptop CPU, ~300+ simulation ticks/sec measured headless in this repo's test environment |
| `detailed` | ~11,200 | proportionally more (still capped by fan-in/fan-out limits in `brain/connectome.py`) | fully synthetic | still interactive; scipy sparse matrix-vector products keep the per-step cost roughly linear in edge count |
| `connectome` | ~20,600 default subset; 139,255 whole brain (FlyWire), or ~25,000 (hemibrain) | ~362,000 subset; 2,700,513 whole brain at ≥5 synapses (FlyWire), or ~20M (hemibrain) | real FlyWire FAFB v783 (auto-downloaded) or your CSV export | subset is interactive on a laptop; whole-brain mode is usable but slower, especially once rendered |

**Why the full connectome is still a heavy target:** the raw FlyWire table
has ~16.85 M `(pre, post, neuropil)` rows, which sum to ~2.70 M weighted
directed edges at the published ≥5-synapse threshold over 139,255 neurons.
A dense recurrent step is a sparse matrix-vector product over those edges,
once per neural integration sub-step, on top of maintaining ~139k neuron
state vectors and their history for visualization (points, colors, edge
subsampling, etc.) every rendered frame. This is not impossible on capable
hardware, but it is well outside what this project's default settings target
("runs on a normal computer", per the project spec). For scale, on the
development machine used for this project, the default 20,600-neuron /
380,394-edge subset processed 10 video frames (3 sub-steps each) in ~0.15 s,
and the whole 139,255-neuron / 2.70 M-edge graph took ~60 ms per frame before
rendering; the 3D viewer, not the dynamics, is the bottleneck at that size.
Recommended paths if you want to push toward `connectome` scale:

1. **Use the default subset, or choose its size.** `--max-neurons` selects
   whole retinotopic tiles (keeping local columnar circuits intact) rather
   than isolated neurons; `--max-neurons 0` is whole brain.
2. **Restrict to a sub-circuit.** For your own CSV exports, use
   neuprint-python / FlyWire's annotations to pull only the optic lobe + a
   handful of downstream ROIs rather than the whole brain.
3. **Reduce the edges drawn in 3D, not the edges simulated.** The
   simulation's sparse dynamics can handle far more edges than the 3D
   viewer should ever try to draw as individual lines;
   `visualization/brain3d.py` already subsamples to `MAX_EDGES_DRAWN`
   edges for rendering while `brain/neuron_model.py` simulates the full
   loaded graph - raise `MAX_EDGES_DRAWN` cautiously.
4. **Lower `--sub-steps` and `--video-fps`.** Neural integration cost
   scales with `steps_per_video_frame`; reducing this trades dynamical
   smoothness for speed.
5. **Move to GPU.** `brain/neuron_model.py` only depends on `scipy.sparse`
   matrix-vector products and elementwise numpy ops - both have direct
   GPU-accelerated equivalents (e.g. `cupyx.scipy.sparse`, or reimplementing
   `LeakyRateNetwork` in PyTorch/JAX using its own sparse or dense
   representations, as flyvis does for the real visual-system-only
   connectome). This is the most promising path to true full-connectome,
   real-time simulation and is one of the main suggestions in `README.md`'s
   future work section.

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
- Maisak, M.S. et al. (2013). *A directional tuning map from the
  connectivity of Drosophila motion detector neurons.* Nature.
  https://doi.org/10.1038/nature12320 (a/b/c/d subtype tuning and the
  preferred-direction map referenced in `brain/network.py`)
- Kamikouchi, A. et al. (2006). *Comprehensive classification of the
  auditory sensory projections in the brain of the fruit fly.*
  Journal of Comparative Neurology 499, 317–356. (JO-A…JO-F classification
  and projections to the AMMC, used to identify auditory receptors in
  `data/flywire_source.py` / `hearing/`)
- Kamikouchi, A., Inagaki, H.K., Effertz, T., Hendrich, O., Fiala, A.,
  Göpfert, M.C. & Ito, K. (2009). *The neural basis of Drosophila
  gravity-sensing and hearing.* Nature 458, 165–171.
- Albert, J.T. & Göpfert, M.C. (2015). *Hearing in Drosophila.*
  Current Opinion in Neurobiology 34, 79–85. (behaviorally relevant
  near-field sound frequencies and courtship-song tuning referenced in
  `hearing/johnstons_organ.py`)
- Eckstein, N., Bates, A.S. et al. (2024). *Neurotransmitter classification
  from electron microscopy images at synaptic sites in Drosophila
  melanogaster.* Cell. (the predicted neurotransmitter labels distributed
  with the FlyWire connectivity release)
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
