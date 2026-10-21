# Wildfire spread inversion

Code, archived numerical outputs and digitised isochrones for the manuscript
*A good fire fit is not enough: Parameter redundancy, wind identifiability and
forecast consequences in wildfire spread inversion*, by Xin Dong, Tuo Yan,
Jun Shen, Chiyuan Li and Tianwei Zhang. Prepared for Ecological Informatics;
no journal acceptance or publication is claimed.

## Reproduce and inspect

```bash
python -m venv .venv
# Activate the environment for your operating system
python -m pip install -r code/requirements.txt
python code/validate_deposit.py
python code/prepare_figure3_records.py --out figure3_source
# Additional independent residual calibration (separate output folder)
python code/residual_calibration.py --out residual_calibration_run
# Full analysis; regenerates and overwrites code/output
python code/run_all.py
```

See [code/README.md](code/README.md) for script purposes, dependencies,
experimental inputs, final decision rules and the distinction between the
original analytical plots and current editable PowerPoint figures.

## Data and figure provenance

`code/output/` contains archived CSV/JSON outputs, including 1,000 labelled,
1,140 graded and 780 additional model-mismatch records. The 2,440 core cases
also include 240 residual-diagnostic and 60 unknown-geometry inversions, whose
summaries are archived but whose paired per-case winds were not retained here.
Current Figure 3b contains the 2,140 archived labelled and graded pairs.

The revised methods paper adds 8,000 independently seeded calibration and
validation cases for the residual screen. They are archived separately as
`code/output/residual_calibration_cases.csv`, with thresholds, exact binomial
intervals, 48 independent optimiser checks and profile comparisons in
`code/output/residual_calibration.json`. These simulations do not change the
original figure coordinates or the core benchmark counts. Calibration averages
over the stated synthetic design and assumes covariance anchored to generating
geometry; it is not a per-fire fitted-model bootstrap. Full profiles can expose
a capped plateau without guaranteeing that a nominal interval includes it.

`code/annaburroo/` contains the seven-fire selection, digitising anchors and
analysis tools. Original maps and wind measurements are available from CSIRO:
[1986 Annaburroo Experimental Grassland Fire Data](https://doi.org/10.25919/ycd3-w209),
CC BY 4.0. Extract the source data to `data_external/annaburroo/raw/`.
The upstream 584 MB archive is not redistributed in this repository.
Without it, the full runner skips the experimental analysis and retains the
archived experimental outputs.

Figures 1 and 2 and Figure 3a are explanatory artwork. Figure 3b is derived
from archived numerical coordinates; Figure 4 contains original CSIRO maps
and digitised overlays. Editable PowerPoint layouts and final figure exports
are supplied separately with the manuscript. Legacy plot numbers in
`make_figures.py` do not refer to these four layouts.

## Checks and limitations

`validate_deposit.py` checks syntax, JSON/CSV records, case counts, selected
coverage results, exact front compensation, the precision-floor identity,
Rothermel reference values and file checksums. These focused checks are not
a fresh run of the full simulation benchmark or an independent validation of
the experimental digitisation. Coverage statements are conditional on the
stated model, covariance and calibration assumptions.

## Licence and citation

Code: MIT, see `LICENSE`. Archived outputs and digitising anchors: CC BY 4.0,
see `LICENSE-data`. Attribute the upstream CSIRO data when using derived
experimental outputs. Citation metadata are in `CITATION.cff`.
`MANIFEST.sha256` records the deposited file contents.
This repository has no assigned DOI. Use a commit permalink to identify the
exact version used in the manuscript.

Claude (Anthropic) assisted analysis-code development. OpenAI Codex (OpenAI)
assisted checking, documentation and manuscript figure preparation. The tools
did not generate primary experimental observations or benchmark values.

## Audit of the original residual fits

`audit_legacy_residual_fits.py` reproduces the 240 original observations and
one-start LM counts, then checks the fit by profiling out the linear amplitude.
Eight old solutions were suboptimal. Correctly specified independent/correlated
cases change from 4/60 and 2/60 residual flags to 1/60 and 0/60, respectively.
The paired refits are archived separately and are not independent new trials.
The original archived values and manuscript figure coordinates are preserved.
