# Analysis scripts and archived results

The scripts implement the equations, synthetic benchmarks and experimental-fire
analysis described in the manuscript. Random seeds are fixed in the scripts.
The original analytical plots and their legacy figure numbers differ from the
four figures in the current manuscript.

## Run and check

```bash
python -m pip install -r code/requirements.txt
python code/validate_deposit.py
python code/prepare_figure3_records.py --out figure3_source
# Full analysis; overwrites numerical files in code/output
python code/run_all.py
```

The recorded analysis environment is Python 3.13.9, NumPy 2.3.5, SciPy 1.16.3
and Matplotlib 3.10.6. Pillow 12.0.0 is specified for map/image processing.
The focused validator reads the archived records and checks selected analytical
identities and published fuel-model reference values. It is not a full rerun.
The full analysis previously took about one hour; hardware affects runtime.

## Script map

| Script | Analysis or output |
|---|---|
| `pyrofield_reduced.py` | Reduced spread model, exact ellipse and observation operators |
| `rothermel.py` | Standard fuel models; run directly for Anderson reference comparisons |
| `shape_law_profiles.py` | Noise-free profiles, calibration bias and shape-law regimes |
| `positional_error_identifiability.py` | Information after nuisance adjustment under correlated positional error |
| `identifiability_extensions.py` | Precision floor, sampling density, unknown ignition and wind periods |
| `guard_benchmark.py` | 1,000 labelled inversions: five classes and two error models |
| `refit_factorial.py` | Sensitivity to refit starting point, iterations and damping |
| `graded_benchmark.py` | 1,140 graded inversions: precision, priors and interval coverage |
| `misspecification_checks.py` | Dilation; 240 residual-diagnostic and 60 unknown-geometry inversions |
| `benchmark_tables.py` | Final bound-aware decisions and conditional coverage summaries |
| `mismatch_scenarios.py` | 780 additional inversions under time-varying wind and heterogeneous fuel |
| `fire_behaviour.py` | Compensation by fuel attributes; intensity conventions and wind limits |
| `forecast_scenarios.py` | Arrival delays under historical/future wind-error relationships |
| `annaburroo/annaburroo_demo.py` | Seven experimental fires, five plots and 13 digitised isochrones |
| `annaburroo/snap_isochrones.py` | Snap recorded anchors to original map lines |
| `annaburroo/grid_view.py` | Inspect map digitisation against source imagery |
| `make_figures.py` | Legacy analytical plots and computational graphical abstract |
| `prepare_figure3_records.py` | Exact 2,140 coordinates and 64-by-64 log-space bin counts for current Figure 3b |
| `validate_deposit.py` | Read-only archived-data and focused numerical checks |
| `run_all.py` | Analyses in dependency order; skips experimental analysis if raw data are absent |

## Decision rule and counts

The final rule in `benchmark_tables.py` labels wind determined when D_U >= 4
at 30% upward log displacement and neither displaced wind nor a nuisance
parameter hits a bound. Otherwise it is unresolved; a would-be determination
with a bound hit is inconclusive. The provisional `GC_threeway` summaries use
an earlier treatment of bounds and must not replace the final table summaries.

The 2,440 core inversions are 1,000 labelled, 1,140 graded, 240 covariance
diagnostic and 60 unknown-bearing/ignition cases. The 780 model-mismatch
inversions are additional. Current Figure 3b pools the 2,140 archived paired
labelled and graded records. This pooled display includes different assumptions
and unidentifiable cases; it is not an overall accuracy or coverage estimate.

## Experimental source data

Download the original CSIRO dataset at https://doi.org/10.25919/ycd3-w209
(CC BY 4.0) and extract it to `data_external/annaburroo/raw/` beside `code/`.
For example, that folder must contain `C04/C04_2mwind.csv` and the source maps
specified in `code/annaburroo/fires.json`. Without raw inputs the experimental
analysis is skipped and its archived outputs remain available.

The repository contains digitising anchors, snapped lines, fitted isochrones
and numerical summaries. It does not contain the upstream 584 MB archive.
The experimental fires are exploratory observations with shared plot records,
not seven independent validations of operational forecast performance.

## Current figure provenance

Figures 1 and 2 and Figure 3a are explanatory artwork. Figure 3b uses the
archived numerical coordinates without jitter, smoothing or invented points.
Figure 4 uses crops of original CSIRO isochrone maps with digitised open lines
overlaid. PowerPoint sources and publication exports accompany the manuscript
submission separately; running `make_figures.py` does not rebuild those layouts.
No image-recovered or AI-generated trace is part of the numerical evidence.
