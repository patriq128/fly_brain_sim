# models/

Placeholder for future trained/fitted parameter sets - e.g. per-cell-type
dynamics parameters imported from `TuragaLab/flyvis`, or optimized synaptic
sign/weight tables derived from real connectome neurotransmitter
predictions. Nothing in the current DEMO/DETAILED/CONNECTOME pipeline
requires files here; `brain/neuron_model.py`'s parameters currently come
from `simulation/config.py` (`NeuronDynamicsParams`) directly.

See "Future work toward biological accuracy" in the top-level README.md,
item 2, for the intended use of this directory.
