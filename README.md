# Morpho-Feature Analysis

A Python pipeline for extracting tissue-scale morphological and mechanical features from cell segmentation images (e.g. produced by [Cellpose](https://github.com/MouseLand/cellpose)). Developed in the [Petridou Laboratory](https://github.com/Petridou-Laboratory/).

## Overview

This toolbox computes the following per-image features:

| Feature | Symbol | Description |
|---|---|---|
| Cell–Cell Contact | CCC | Fraction of cell perimeter in direct contact with neighbouring cells |
| Fluid fraction | — | Area fraction occupied by extracellular fluid pockets |
| Contact angle | α | Geometric angle at fluid-pocket vertices; related to surface tension balance |
| Multi-/Tri-cellular junctions | MCJ / TCJ | Counts of vertices shared by ≥3 or exactly 3 cells |
| Average connectivity | ⟨k⟩ | Mean number of rigid contacts per cell (normalised) |
| Giant Rigid Cluster | GRC | Fraction of cells belonging to the largest mechanically rigid cluster |

Rigidity is assessed using the **(2,3)-pebble game** algorithm for 2D constraint networks.

## The Pebble Game Algorithm

The rigidity analysis (`getClusters` in `analysis_functions.py` and the standalone `pebble.py` module) is based on the **pebble game** algorithm:

> Jacobs, D. J., & Hendrickson, B. (1997). An algorithm for two-dimensional rigidity percolation: The pebble game. *Journal of Computational Physics*, **137**(2), 346–365. https://doi.org/10.1006/jcph.1997.5809

The Python implementation of the pebble game (`pebble.py`) was originally written by **Leyou Zhang** (2016) as a translation from the original Fortran code. It was subsequently **adapted and adjusted** for this biological imaging context by collaborator **Bernat Corominas-Murtra**.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Petridou-Laboratory/morpho-feature-analysis.git
cd morpho-feature-analysis
pip install numpy pandas scikit-image scipy networkx matplotlib tifffile tqdm
```

## Repository Contents

```
analysis_functions.py   # Main analysis pipeline
pebble.py                     # (2,3)-pebble game rigidity solver
tutorial.ipynb                # Step-by-step usage tutorial
segmentations/                # Place your segmentation images here
```

## Requirements

```
numpy
pandas
scikit-image
scipy
networkx
matplotlib
tifffile
tqdm
```

## Input Data

Place segmentation images in the `segmentations/` directory. The expected naming conventions are:

- **Experimental data:** `[image_name]_cp_masks.png` (labelled segmentation from Cellpose)
- **Optional fluorescence mask:** `[image_name]_IF_binary.tif` (binary fluid channel; same base name as the segmentation)
- **Simulated data:** after renaming, files follow `N[iter]_alpha[0p…]_frac[0p…]_cp_masks.png`

All label values must be positive integers; the background must be 0.

## Usage

Open `tutorial.ipynb` for a complete, annotated walkthrough. The core entry point is `parameter_analysis()` in `analysis_functions.py`:

```python
import analysis_functions as af

results_per_cell, results_per_pocket, adj_matrix = af.parameter_analysis(
    labeled_image_path="segmentations/my_image_cp_masks.png",
)
```

The function returns:
- **`results_per_cell`** – `pd.DataFrame` with one row per cell (area, CCC, MCJ, TCJ, ⟨k⟩, GRC membership, …)
- **`results_per_pocket`** – `pd.DataFrame` with one row per fluid pocket (area, α, …)
- **`adj_matrix`** – adjacency matrix (`numpy.ndarray`) of the cell contact network

## Output

Results are written to two CSV files per image and an `.npy` adjacency matrix, organised under:

```
segmentations/results/per_cell_csvs/
segmentations/results/per_pocket_csvs/
segmentations/results/Adjmatrix/
```

## Citation

If you use this pipeline in your research, please cite the pebble game algorithm:

```
Jacobs, D. J., & Hendrickson, B. (1997).
An algorithm for two-dimensional rigidity percolation: The pebble game.
Journal of Computational Physics, 137(2), 346–365.
https://doi.org/10.1006/jcph.1997.5809
```

## Acknowledgements

- Pebble game Python implementation originally by **Leyou Zhang** (2016).
- Adapted for this biological imaging pipeline by **Bernat Corominas-Murtra**.
- Pipeline developed in the **Petridou Laboratory**.
