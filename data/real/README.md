# data/real — downloaded real connectome data

Nothing large is committed here. `python main.py --level connectome`,
`python main.py --fetch-connectome`, or `python -m data.flywire_source`
downloads and caches:

| File | Size | Source |
|---|---|---|
| `flywire_v783/annotations.tsv` | 27 MB | Schlegel et al. 2024, *Nature*, Supplementary Data 5 |
| `flywire_v783/proofread_connections_783.feather` | 852 MB | FlyWire FAFB v783, Zenodo `10.5281/zenodo.10676866` |
| `flywire_v783/graph_*.npz` | 2–20 MB | built-graph cache (regenerable) |

The files are **CC-BY-4.0**. If you publish results derived from them, cite:

- Dorkenwald, S. et al. (2024). *Neuronal wiring diagram of an adult brain.*
  Nature. <https://doi.org/10.1038/s41586-024-07558-y>
- Schlegel, P. et al. (2024). *Whole-brain annotation and multi-connectome
  cell typing of Drosophila.* Nature.
  <https://doi.org/10.1038/s41586-024-07686-5>
- Eckstein, N., Bates, A.S. et al. (2024) for the neurotransmitter
  predictions distributed with the FlyWire connectivity release.

Metadata for the exact files used is written to
`flywire_v783/PROVENANCE.json` on first download. Each file is verified
against the published size and checksum; a matching `<file>.verified` marker
records the checksum so later runs skip re-hashing. Use `--connectome-cache`
to put the ~880 MB cache on another disk, and `--no-download` to fail fast
instead of downloading. See
[`docs/SIMULATION_NOTES.md`](../../docs/SIMULATION_NOTES.md) for exactly
which parts of the resulting simulation are measured data and which are
model assumptions.
