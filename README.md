# Radial bright-band detection

Hasan Hosseini
Swedish Meteorological and Hydrological Institute (SMHI), Sweden  
Email: hasan.hosseini@smhi.se

This repository implements the radial bright-band (BB) detection and significant rise–fall segmentation methodology described in Hosseini et al. (2026).

It processes ODIM HDF5 radar volumes and generates illustrations of azimuthal reflectivity profiles and BB detections over the azimuth–height domain, corresponding to Figures 4 and 5 of the manuscript. The results are also saved as CSV files containing BB heights, significant rise–fall segment boundaries, and other diagnostics, including the stratiformity check across azimuths.

## Installation

Clone the repository:

```bash
git clone https://github.com/hasan-hosseini/radial-bright-band.git
cd radial-bright-band
```

Install the required Python packages:

```bash
python3 -m pip install -r requirements.txt
```

## Generate results for Figures 4 and 5

Supply the original radar volume using `--input`. Place the two original HDF5 files corresponding to Figures 4 and 5 in `inputs/`.

**Figure 4 — Ängelholm C-band Weather Radar (CWR), 4 February 2022, 14:00 UTC**

The following command generates `fig4a_straight.png`, `fig4a_straight.csv`, `fig4b_straight.png` and `fig4b_straight.csv` in `outputs/`.

```bash
python3 reproduce.py \
  --input inputs/seang_qcvol_pn129_20220204T140000Z_0x81540b.h5 \
  --both --tag fig4
```

**Figure 5 — Ängelholm CWR, 18 May 2025, 11:50 UTC**

The following command generates `fig5a_straight.png`, `fig5a_straight.csv`, `fig5b_straight.png` and `fig5b_straight.csv` in `outputs/`.

```bash
python3 reproduce.py \
  --input inputs/seang_qcvol_pn129_20250518T115000Z_0x81148b.h5 \
  --both --tag fig5
```

The default propagation method is `straight` (straight-line spherical-Earth geometry). To use standard refraction, add `--propagation standard`; the paired output filenames then end in `_standard` instead of `_straight`. In single-panel mode, `--tag` sets the filename stem directly.

Panel a uses maximum Z and panel b uses the 90th percentile of Z in each azimuth–height bin. To generate one panel with a customized percentile, replace `--both --tag fig5` with, for example, `--percentile 95 --tag fig5_p95`. Specify `--output-dir` to change the output location. Run `python3 reproduce.py --help` for all options.

## Scripts

| File | Purpose |
|---|---|
| `reproduce.py` | Main entry point: prepares reflectivity profiles, produces the figures and exports the CSV diagnostics. |
| `bb_core.py` | Implements the radial BB detection methodology, including azimuth–height reflectivity gridding (`profile_grid`), raw BB characteristic detection and segment selection (`raw_components`), significant rise–fall segmentation (`ischange_gradient`), outlier filtering and smoothing (`mask_and_smooth`), and triangular azimuthal smoothing (`smooth_heights_triangular`). |
| `plot_original.py` | Generates azimuth–height illustrations of linear reflectivity (Z), accepted smoothed BB peaks and boundaries, and significant rise–fall segment boundaries (`plot_Z_azimuth_height_summary_ny_final`), corresponding to Figures 4 and 5 of the manuscript. |
| `requirements.txt` | Python dependencies. |

## Reflectivity and BB detection

The default BB analysis grid has 360 azimuth centers (0.5°, 1.5°, …, 359.5°), and MRR-aligned height bins from 356 to 2386 m a.s.l. at 70 m intervals. The smoothing sector is 6° wide. Reflectivity is processed in linear Z (mm⁶ m⁻³); ODIM `nodata` is excluded and `undetect` is represented by −40 dBZ before conversion to linear Z.

A smoothed BB peak is accepted when it lies within the smoothed main-gradient bounds or a smoothed significant rise–fall segment, with the relevant heights inside the displayed height interval. BBB and BBT are reported as final heights only when the main-gradient bounds are accepted. A segment-supported peak can be accepted even when those bounds are unavailable.

The stratiformity check is reported separately for each BB characteristic using the number of accepted azimuths (at least 30) and the height coefficient of variation (at most 0.25). These thresholds can be changed with `--stratiform-min-radials` and `--stratiform-max-cov`. Stratiformity is reported as a diagnostic and does not change the figure acceptance mask.

## Plot settings

The default figure window is 300–3000 m a.s.l. The reflectivity plotting grid is aligned to the MRR reference origin (76 m a.s.l.) with 70 m spacing, and its lower extent respects radar altitude read from ODIM `/where/height`. The plotting grid is extended independently of the BB analysis grid to cover the display window. `--ylim`, `--mrr-reference-altitude`, `--height-min`, `--height-max`, and `--height-step` control the respective settings. The default analysis limits reproduce the specified Ängelholm configuration; use appropriate analysis limits for other sites.

Both panels use a shared linear color scale in `--both` mode. The default minimum is zero; the common maximum is the larger of the 99th percentiles of the two plotted Z fields. Values above the maximum are saturated. Colorbar ticks are integers in the displayed ×10ⁿ units. Use `--vmin` and `--vmax` to specify fixed limits, for example:

```bash
python3 reproduce.py --input /path/to/volume.h5 --both --tag fig5 --vmin 0 --vmax 5000
```

## CSV columns

Each CSV contains 1080 (= 3 × 360) rows sorted by UTC datetime, azimuth and characteristic (BBB, BBP, BBT). All heights are in metres above sea level. Missing numeric values are blank.

| Column | Description |
|---|---|
| `datetime_utc`, `radar`, `azimuth_deg`, `bb_characteristic` | Observation time, radar identifier, regular azimuth and BB characteristic (bottom, peak, top). |
| `raw_height_m_asl` | Unsmoothed BB characteristic. |
| `smoothed_height_m_asl` | Final accepted smoothed BB characteristic; blank if not accepted. |
| `raw_all_segment_bounds_m_asl` | All raw significant rise–fall segments, as a JSON list of [lower, upper] pairs. |
| `selected_segment_index` | Zero-based index into that list for the segment selected using the raw peak; blank when none is selected. |
| `smoothed_segment_lower_m_asl`, `smoothed_segment_upper_m_asl` | Smoothed boundaries of the selected segment. |
| `segment_fallback` | 0: accepted using main-gradient bounds; 1: BBP accepted using a significant rise–fall segment because main-gradient bounds were not accepted; blank: no accepted height for this characteristic. |
| `outlier_01` | 1 if any raw BB characteristic at the azimuth was flagged as an outlier before smoothing; 0 otherwise. |
| `stratiform_01`, `stratiform_n_valid`, `stratiform_cov` | Stratiformity flag, valid azimuth count and height coefficient of variation for the characteristic. |
| `status` | Final acceptance outcome or preliminary detection diagnostic (see status definitions below). |
| `n_scans`, `input_file`, `reflectivity_percentile` | Number of elevation scans, input filename and reflectivity aggregation percentile. |

The four segment-specific columns are populated only on BBP rows, including when the main-gradient bounds are accepted. A BBP row with no raw segments has an empty JSON list (`[]`); smoothing can still supply segment heights from adjacent azimuths. The corresponding BBB and BBT cells are blank. Raw selected segment boundaries can be obtained from the list using the selected index. Smoothed segment boundaries are reported before acceptance masking for diagnostic purposes.


### Status definitions

| Status | Description |
|---|---|
| `accepted_main` | Final smoothed height accepted using the main-gradient bounds. |
| `accepted_segment_fallback` | Final smoothed BBP accepted using the significant rise–fall segment because the main-gradient bounds failed acceptance. |
| `outlier_unaccepted` | At least one raw BB characteristic was flagged as an outlier at the azimuth, and the characteristic in this row has no accepted smoothed height. |
| `lowest_level_below_threshold` | The lowest analysis level has missing reflectivity or linear Z below the threshold (default 1 mm⁶ m⁻³). No accepted smoothed height is available for this row. |
| `too_few_wet_levels` | Fewer than the required number of finite levels exceed the reflectivity threshold (default 5). No accepted smoothed height is available for this row. |
| `missing_vertical_levels` | At least one analysis level remains missing after profile gridding and isolated-gap interpolation. No accepted smoothed height is available for this row. |
| `candidate_unaccepted` | The raw profile passed the preliminary checks, but no final smoothed height was accepted for this characteristic. For BBP, neither valid smoothed main-gradient bounds nor a valid smoothed significant rise–fall segment encompasses the smoothed peak. For BBB and BBT, the main-gradient bounds failed acceptance. Required heights may be missing or outside the displayed height interval. |
| `invalid_profile` | Initial internal raw-profile status; replaced by one of the more specific raw-profile statuses during normal processing and not expected in a completed CSV. |

Accepted statuses take precedence over raw-profile diagnostics. Smoothing may reconstruct heights from neighboring azimuths even when the raw profile was rejected or flagged as an outlier. The `outlier_01` flag retains that information independently of final acceptance.
