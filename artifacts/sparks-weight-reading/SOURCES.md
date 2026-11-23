# Source networks and generated maps

These maps and numerical results derive from the networks released by Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab for *Sparks of Weight-Space Programming*.

- [Original article](https://rikiyat.github.io/blog/weight-space-programming/)
- [Source experiment](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp1_which_game)
- [Upstream license at the pinned revision](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/LICENSE)
- [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

The source networks are distributed under CC BY 4.0. Changes in this derived material: compute input-channel L2 norms, render heatmaps, and evaluate a consistent hidden-unit permutation. The original network files are downloaded to the user’s cache and are not included here.

`norms.json` contains all four exact source URLs and SHA-256 hashes. The tool reads the released full-precision arrays. Figure row names follow the upstream game labels.

Recreate the outputs with `python weight_reading.py sparks --output results/weight-reading` from the repository root. The independently written tool is `weight_reading.py`; this source notice covers the upstream network data and its derived maps.
