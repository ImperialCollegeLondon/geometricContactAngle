# geometricContactAngle
Python toolkit for measuring **contact angles in 3D porous media** from **segmented** (three‑phase: Fluid 1, Fluid 2, Solid) images. The toolkit provides end‑to‑end analysis via two workflows:<br>
1. **Geometric measurement** (`Main.py`)<br>Identifies three‑phase contact lines and computes local contact angles. It exports overall statistics (mean, standard deviation, and number of contact loops), per‑loop summaries, and per‑point coordinates (x, y, z) with contact‑angle values. *Optional plotting is available via* `plotting.py`, which generates a histogram with the mean and the standard deviation.
2. **Spatial interpolation** (`Main_SpatialInterpolation.py`)<br>Transforms discrete measurements into a continuous 3D contact‑angle field, producing a voxel‑wise contact‑angle map (a 3D TIFF of the pore space), statistical visualizations (including distributions and wettability classes), and metadata that documents the analysis parameters and results.
## Python version & dependencies
- **Python**: 3.9+
- **Core deps**:
  - `NumPy`
  - `SciPy`
  - `scikit-image`
  - `trimesh`
  - `tifffile`
  - `matplotlib`
  - `pandas`
  - `statsmodels`
  - `tqdm`<br>
**Quick install (pip):**<br>`pip install NumPy SciPy scikit-image trimesh tifffile matplotlib pandas statsmodels tqdm`
## Demo
- An example 3D TIFF image is included with dimensions **300 x 300 x 250** Run `Main.py` with the correct TIFF filename and the ***correct phase label index***; the output is a text file containing contact‑angle measurements. To generate the histogram distribution, run `plotting.py`.<br>
- Then run `Main_SpatialInterpolation.py` to produce the spatial wettability information and a 3D TIFF of the pore space in which voxel values represent the contact angle of the pore surfaces.
## Paper
If you use this work please cite:<br>  F. Aljaberi, H. Belhaj, S. Foroughi, M. Al-Kobaisi, M. Blunt, Spatially Distributed Wettability Characterization in Porous Media, (2025). https://doi.org/10.48550/arXiv.2507.01617.
 
