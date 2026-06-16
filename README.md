### Modifications from the Original NoMaD Codebase

This repository is based on the original NoMaD/ViNT implementation and includes several modifications for robot docking research.
NoMaD: https://github.com/robodhruv/visualnav-transformer

### Main Changes
- Applied max-normalization to distance labels.
- Replaced waypoint prediction with velocity prediction.
- Added direct support for HDF5-based datasets without converting trajectories into image-folder structures.
- Added evaluation and visualization tools for position error, and heading error analysis.
- Adapted the training and inference pipeline.
- Only used docking dataset. (not Init)

### Training
Run training or test at the train folder:

```bash
export PYTHONPATH=/home/<username>/.../visualnav-transformer/diffusion_policy:$PYTHONPATH
```

```bash
python ./train.py -c ./config/nomad.yaml
```
