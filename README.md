# Fly Brain Sim

An experimental, open-source Python project that simulates how visual
information from a video you provide might propagate through a
**simplified, connectome-inspired model of the Drosophila melanogaster
visual system and brain**, with a live interactive 3D visualization of the
resulting neural activity.

> **This is a computational simulation inspired by fly neuroscience, not a
> reproduction of a real fly's brain, thoughts, or consciousness.** Every
> stage of the model is labeled in the code and in
> [`docs/SIMULATION_NOTES.md`](docs/SIMULATION_NOTES.md) as one of:
> real **biology**, real **connectome** data, a generic **simulation**
> technique, or a project-specific **assumption**. Please read that document
> before citing anything from this project as biologically accurate.

![status](https://img.shields.io/badge/status-research%20prototype-orange)
![python](https://img.shields.io/badge/python-3.10%2B-blue)

---

## What this actually does

1. You give it a video file.
2. Frames are extracted and downsampled onto a grid standing in for the
   fly's ommatidial array (`vision/retina.py`).
3. That signal is pushed through a simplified feedforward pathway whose
   *stage order* mirrors the real fly optic lobe: retina → lamina →
   medulla → lobula / lobula plate → central complex / mushroom body
   (`brain/regions.py`, `brain/connectome.py`).
4. Local motion is detected with a vectorized **Hassenstein-Reichardt
   correlator** (the same mathematical model believed to underlie the
   fly's real T4/T5 direction-selective neurons) (`brain/motion_pathway.py`).
5. All neurons are simulated as a single large recurrent network of
   generic leaky-rate units (`brain/neuron_model.py`), driven by the
   vision pathway and by each other, at a controllable sub-step rate.
6. An interactive **PyVista 3D scene** shows every neuron as a point,
   colors it by which brain region it belongs to and how active it
   currently is, draws (a subsample of) its synapses, and updates live as
   the video plays (`visualization/brain3d.py`).
7. A synced video panel and a live statistics readout (active neuron
   count, per-region activity, firing-rate estimate, strongest active
   connections, sim time, FPS) run alongside it
   (`visualization/video_panel.py`, `visualization/stats_panel.py`).
8. You can play, pause, seek, and restart at any time.

## Why it's built this way (research background)

This project is directly inspired by a recent line of work that simulates
or reconstructs neural activity **from real connectome structure**, rather
than training an arbitrary neural network and calling it a brain model:

- **FlyWire / FAFB**: a complete adult Drosophila brain connectome,
  ~139,000 neurons and ~5×10⁷ synapses, released as an open, community-
  annotated resource (Dorkenwald et al. 2024, *Nature*,
  https://doi.org/10.1038/s41586-024-07558-y; Schlegel et al. 2024,
  *Nature*, https://doi.org/10.1038/s41586-024-07686-5).
- **Hemibrain**: an earlier, still widely used dense reconstruction of
  ~25,000 central-brain neurons and ~20 million synapses (Scheffer et al.
  2020, *eLife*; https://www.janelia.org/project-team/flyem/hemibrain),
  queryable via neuPrint.
- **flyvis** (Lappalainen et al. 2024, *Nature*,
  https://doi.org/10.1038/s41586-024-07939-3,
  https://github.com/TuragaLab/flyvis): a "deep mechanistic network" whose
  *architecture* is fixed by real connectome measurements of 64 optic-lobe
  cell types, with only per-neuron/synapse dynamical parameters learned by
  optimizing a motion-estimation task — and the result reproduces real,
  experimentally measured T4/T5 direction selectivity. This is the direct
  template for this project's philosophy: **wire the model the way the
  real circuit is wired, then let dynamics run on top of that structure**,
  rather than "just train a CNN on the video and call it a fly."
- **NeuroMechFly / FlyGym** (Ramdya lab, EPFL;
  https://pypi.org/project/flygym): a full embodied biomechanical fly
  simulation with a Gym/RL interface. This project does not attempt body
  or motor simulation, but it's the natural next step (see Future Work).

Full citation list: [`docs/SIMULATION_NOTES.md`](docs/SIMULATION_NOTES.md#4-sources-cited-in-this-project).

## Quick start

```bash
git clone <this-repo>
cd fly_brain_sim
pip install -r requirements.txt

# Try it immediately with an auto-generated synthetic test video:
python main.py

# Or with your own video:
python main.py --video path/to/your_clip.mp4

# Larger synthetic network:
python main.py --video path/to/your_clip.mp4 --level detailed

# Headless sanity check (prints network size, no windows):
python main.py --level demo --summary-only
```

Two windows open: a **3D brain view** (PyVista) and a **video / photoreceptor
panel** (OpenCV). Controls (work in either window):

| Key | Action |
|---|---|
| `space` | play / pause |
| `r` | restart |
| `←` / `→` | seek back / forward ~1s |
| `q` / `Esc` | quit |

## Simulation levels

| Level | Neurons | Data source | Notes |
|---|---|---|---|
| `demo` (default) | ~1,400 | synthetic, connectome-inspired | runs comfortably on a laptop CPU |
| `detailed` | ~15,000 | synthetic, connectome-inspired | still interactive; more columns/cell density |
| `connectome` | up to real dataset size | **real data you supply** | see below - not recommended at full whole-brain scale for real-time use |

```bash
python main.py --video clip.mp4 --level connectome --connectome-path ./data/my_hemibrain_export
```

`--connectome-path` must contain `neurons.csv` and `edges.csv`. Full
instructions for exporting these from neuPrint (hemibrain) or FlyWire are
documented in [`data/connectome_loader.py`](data/connectome_loader.py) —
real connectome data is not, and cannot be, bundled in this repository
(size and data-use licensing). See
[`docs/SIMULATION_NOTES.md`](docs/SIMULATION_NOTES.md#3-simulation-levels-and-their-computational-cost)
for the compute-cost reasoning behind these tiers.

## Project structure

```
fly_brain_sim/
├── main.py                    # CLI entry point
├── requirements.txt
├── README.md
├── docs/
│   └── SIMULATION_NOTES.md    # full biology/connectome/simulation/assumption breakdown + cost analysis
├── vision/
│   ├── retina.py               # photoreceptor sampling model
│   └── frame_source.py         # video I/O, play/pause/seek/restart
├── brain/
│   ├── regions.py               # brain region metadata + evidence labels
│   ├── connectome.py            # synthetic + real connectome graph construction
│   ├── motion_pathway.py        # Reichardt-correlator motion detection (T4/T5-inspired)
│   ├── neuron_model.py          # vectorized leaky-rate neuron dynamics
│   └── network.py               # wires vision + connectome + dynamics together
├── simulation/
│   ├── config.py                 # SimLevel enum, region sizes, dynamics params
│   └── engine.py                  # main tick loop / transport controls
├── visualization/
│   ├── brain3d.py                # PyVista interactive 3D scene
│   ├── video_panel.py             # OpenCV video + photoreceptor-grid panel
│   └── stats_panel.py             # live stats text formatting
├── ui/
│   └── app.py                     # ties engine + visualizers + keyboard controls together
├── data/
│   ├── connectome_loader.py       # real-data loader + schema/download docs
│   └── sample/make_sample_video.py # synthetic test-video generator
├── models/                        # placeholder for future trained/fitted parameter sets
└── tests/                         # pytest unit tests for dynamics, connectome, vision
```

The architecture is intentionally modular at every seam named above so any
one piece can be swapped for a more biologically accurate version without
touching the rest — see Future Work.

## Running the tests

```bash
pip install pytest
pytest tests/ -v
```

## Future work toward biological accuracy

Roughly in order of impact per effort, per the honesty framework in
`docs/SIMULATION_NOTES.md`:

1. **Real connectome by default.** Wire up an automatic neuprint-python
   fetch for a bounded ROI set (optic lobe + central complex) so
   `--level connectome` works out of the box for anyone with a free
   neuPrint token, instead of requiring a manual CSV export.
2. **Replace the generic leaky-rate neuron with flyvis's dynamics.**
   `TuragaLab/flyvis` already provides connectome-constrained, *trained*
   per-cell-type dynamics for the optic lobe motion pathway — swapping
   `brain/neuron_model.py`'s dynamics for a loaded flyvis model on the
   optic-lobe subgraph would replace our biggest "SIMULATION, not
   biology" gap with something empirically validated.
3. **Neurotransmitter-aware signs.** Use FlyWire/hemibrain's predicted
   neurotransmitter identity per neuron (available in both datasets) to
   set excitatory/inhibitory sign per edge, instead of this project's
   "excitatory unless you tell us otherwise" default in
   `data/connectome_loader.py`.
4. **Spiking dynamics.** Swap the rate model for a leaky
   integrate-and-fire (or conductance-based) model with actual spike
   trains, enabling spike-timing-based statistics and STDP-style
   plasticity experiments.
5. **Real 3D morphology.** Pull actual skeleton/mesh coordinates from
   neuPrint/FlyWire instead of the schematic layout in `brain/regions.py`,
   for a geometrically accurate (not just topologically labeled) 3D scene.
6. **Embodiment.** Couple this visual/brain model's output to
   NeuroMechFly/FlyGym so simulated "decisions" actually drive a simulated
   fly body, closing the sensorimotor loop this project currently leaves
   open (central complex/mushroom body activity here is a dead end,
   displayed but not acted upon).
7. **GPU acceleration.** Port `LeakyRateNetwork` to a GPU sparse backend
   to make whole-connectome (`~139k` neuron) real-time simulation
   practical - see `docs/SIMULATION_NOTES.md §3` for specifics.

## License

Choose and add a license appropriate for your use (e.g. MIT/BSD for the
code). Note that any *real* connectome data you load via `--connectome-path`
carries its own license/attribution requirements from FlyWire or Janelia
(hemibrain is CC-BY) - cite the source papers listed in
`docs/SIMULATION_NOTES.md` if you publish results derived from real data.
