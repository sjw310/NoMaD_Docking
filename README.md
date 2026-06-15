### Modifications from the Original NoMaD Codebase

This repository is based on the original NoMaD/ViNT implementation and includes several modifications for robot docking research.

### Main Changes
- Applied max-normalization to distance labels.
- Replaced waypoint prediction with velocity prediction.
- Added direct support for HDF5-based datasets without converting trajectories into image-folder structures.
- Added evaluation and visualization tools for position error, and heading error analysis.
- Adapted the training and inference pipeline.
