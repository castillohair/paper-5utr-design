# Optimizing 5’UTRs for mRNA-delivered gene editing using deep learning
This repository contains data analysis and sequence design code in the publication "Optimizing 5’UTRs for mRNA-delivered gene editing using deep learning", including code to generate all Figure panels.

## Contents

- `analysis_cell_type`: Code for analysis of cell type data in Figure 1 and related Supplementary Figures.
- `analysis_random_end`: Code for analysis of the random-end MPRA libraries from Figure 3 and related Supplementary Figures.
- `megatal_5utr_design`: Notebooks and scripts for 5'UTR design, used in megaTAL gene editing assays.
- `megatal_gene_editing_analysis`: Code for analysis of megaTAL gene editing results
- `models_sample2019`: Deep learning models from Sample, et al. Nat. Biotech 2019. must be downloaded and placed here for the code in this repo to work. See README file inside.
- `polysome_profiling_data`: Polysome profiling data acquired in this study must be placed here for the code in this repo to work. See README file inside.
- `polysome_profiling_sample2019`: Polysome profiling data acquired in Sample, et al. Nat. Biotech 2019. must be downloaded and placed here for the code in this repo to work. See README file inside.

## Requirements
All code was written in Python 3. With the exception of the sequence design code (`megatal_5utr_design`), the following package versions were used:
- `matplotlib` 3.5.1
- `numpy` 1.22.1
- `pandas` 1.4.3
- `scikit-learn` 1.0.2
- `scipy` 1.7.3
- `seaborn` 0.12.2
- `tensorflow` 2.7.0

For sequence design, the following are required:
- [`isolearn`](https://github.com/johli/isolearn/) 0.2.1
- [`Fast SeqProp`](https://github.com/johli/seqprop/) 0.1
- [`DEN`](https://github.com/johli/genesis/) 0.1
- `keras` 2.2
- `tensorflow` 1.15

All standard packages can be installed with `pip` or `conda`. For `isolearn`, `Fast SeqProp`, and `DEN`, follow the instructions in the respective repos.

## Usage
Most of the analysis code is in Jupyter notebooks, each of which can be opened and run to reproduce a specific analysis or design task.
