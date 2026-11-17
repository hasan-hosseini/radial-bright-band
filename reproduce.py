#!/usr/bin/env python3
"""
Main script for radial bright-band (BB) detection and generation
of illustrations corresponding to Figures 5 and 6 of Hosseini et al. (2026).

Processes ODIM HDF5 radar volumes and generates azimuthal reflectivity
plots and CSV files containing BB characteristics, significant rise–fall
segments, and stratiformity diagnostics.

Author: Hasan Hosseini
Affiliation: Swedish Meteorological and Hydrological Institute (SMHI)
"""

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import h5py
from bb_core import profile_grid, raw_components, mask_and_smooth
from plot_original import plot_Z_azimuth_height_summary_ny_final

AZ = np.arange(360, dtype=float) + 0.5

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input', required=True, type=Path, help='Path to one ODIM HDF5 radar volume')
    p.add_argument('--output-dir', type=Path, default=Path('outputs'))
    p.add_argument('--tag', default=None, help='Output stem: fig5a for single mode or fig5 for --both')
    p.add_argument('--both', action='store_true', help='Generate panels a (100th percentile) and b (90th percentile), each with a CSV')
    p.add_argument('--percentile', type=float, default=100, help='Azimuth/height bin percentile of linear Z')
    p.add_argument('--height-min', type=float, default=356)
    p.add_argument('--height-max', type=float, default=2386)
    p.add_argument('--height-step', type=float, default=70)
    p.add_argument('--mrr-reference-altitude', type=float, default=76,
                   help='MRR-aligned vertical-grid origin (m a.s.l.) used in the original height-grid script')
    p.add_argument('--sector', type=int, default=6, help='Full width of circular triangular smoothing sector, degrees')
    p.add_argument('--z-threshold', type=float, default=1, help='Threshold in linear Z')
    p.add_argument('--min-valid-bins', type=int, default=5)
    p.add_argument('--gradient-percentile', type=float, default=85)
    p.add_argument('--max-changes', type=int, default=6)
    p.add_argument('--outlier-zscore', type=float, default=2.5)
    p.add_argument('--propagation', choices=('straight','standard'), default='straight')
    p.add_argument('--quantity', default='DBZH')
    p.add_argument('--min-scans', type=int, default=10)
    p.add_argument('--stratiform-min-radials', type=int, default=30)
    p.add_argument('--stratiform-max-cov', type=float, default=0.25)
    p.add_argument('--ylim', type=float, nargs=2, default=(300,3000), metavar=('MIN','MAX'),
                   help='Display height limits in m a.s.l. (independent of BB detection heights)')
    p.add_argument('--vmin', type=float, default=0)
    p.add_argument('--vmax', type=float, default=None, help='Linear Z maximum; default is the 99th percentile of displayed-height Z (shared when --both)')
    p.add_argument('--dpi', type=int, default=300)
    args=p.parse_args()
    if not args.input.is_file(): p.error(f'Input file does not exist: {args.input}')
    if not (0 <= args.percentile <= 100): p.error('--percentile must be 0..100')
    if args.height_step <= 0 or args.sector < 2 or args.sector % 2: p.error('Invalid height step or smoothing sector')
    if abs((args.height_min - args.mrr_reference_altitude) / args.height_step - round((args.height_min - args.mrr_reference_altitude) / args.height_step)) > 1e-7:
        p.error('--height-min must lie on the MRR-aligned vertical grid')
    return args

def accepted_geometry(processed, y0, y1):
    p=processed['peak']; lo=processed['lower']; hi=processed['upper']
    slo=processed['seg_lower']; shi=processed['seg_upper']
    peak_ok=np.isfinite(p)&(p>=y0)&(p<=y1)
    main=peak_ok&np.isfinite(lo)&np.isfinite(hi)&(lo>=y0)&(hi<=y1)&(lo<=p)&(p<=hi)
    fallback=peak_ok&np.isfinite(slo)&np.isfinite(shi)&(slo>=y0)&(shi<=y1)&(slo<=p)&(p<=shi)
    return main, fallback, (main|fallback)

def plot_heights(a):
    """Build an MRR-aligned plotting grid, respecting the radar-site altitude.

    The original height-grid script uses centers 76 + n*70 m a.s.l.
    The BB analysis grid is independent and remains unchanged.
    Do not derive height centers from --ylim: changing the figure crop must
    not shift the reflectivity sampling levels.
    """
    step = float(a.height_step)
    origin = float(a.mrr_reference_altitude)
    with h5py.File(a.input, 'r') as f:
        radar_height = float(f['where'].attrs['height'])

    # Lowest MRR-aligned center at or above the radar-site altitude.
    first_above_radar = max(0, int(np.ceil((radar_height - origin) / step - 1e-10)))

    # Include the cell covering the requested lower display edge if it is
    # physically above the radar. Otherwise the unmeasured strip remains blank.
    first_for_display = int(np.floor((a.ylim[0] + step / 2 - origin) / step))
    first_index = max(first_above_radar, first_for_display, 0)

    # Include enough cells for the upper display edge.
    last_index = max(first_index, int(np.ceil((a.ylim[1] - step / 2 - origin) / step)))
    heights = origin + step * np.arange(first_index, last_index + 1, dtype=float)
    print(f'Radar altitude {radar_height:g} m a.s.l.; '
          f'MRR plotting centers {heights[0]:g}–{heights[-1]:g} m a.s.l.')
    return heights


def prepare_one(a):
    # BB detection and smoothing always use the scientific analysis height range.
    h=np.arange(a.height_min, a.height_max+a.height_step*0.1, a.height_step)
    if len(h) < 3: raise ValueError('At least 3 height levels required')
    matrix, n_scans = profile_grid(a.input,h,a.percentile,a.propagation,a.quantity)
    if n_scans < a.min_scans: raise ValueError(f'Only {n_scans} scans; need {a.min_scans}')
    raw=[raw_components(p,h,a.z_threshold,a.min_valid_bins,a.gradient_percentile,a.max_changes) for p in matrix]
    processed=mask_and_smooth(raw,a.sector,a.outlier_zscore)
    # Separate reflectivity grid extends to the displayed upper height (3000 m
    # by default). Computing it separately avoids changing BB detections via
    # different vertical endpoint handling/interpolation.
    hp=plot_heights(a)
    plot_matrix,_=profile_grid(a.input,hp,a.percentile,a.propagation,a.quantity)
    hsel=(hp >= a.ylim[0]-a.height_step/2)&(hp <= a.ylim[1]+a.height_step/2)
    valid=plot_matrix[:,hsel]
    valid=valid[np.isfinite(valid)]
    if valid.size == 0: raise ValueError('No valid reflectivity in displayed height range')
    p99=float(np.percentile(valid,99))
    return h, raw, processed, n_scans, hp, plot_matrix, p99


def run_one(a, prepared=None, shared_vmax=None):
    if prepared is None:
        prepared=prepare_one(a)
    h,raw,processed,n_scans,plot_h,plot_matrix,p99=prepared
    main_ok, seg_ok, peak_ok=accepted_geometry(processed,*a.ylim)
    main_bounds=np.column_stack([processed['lower'],processed['upper']])
    seg_bounds=np.column_stack([processed['seg_lower'],processed['seg_upper']])
    vmax=(shared_vmax if shared_vmax is not None else
          a.vmax if a.vmax is not None else p99)
    if not a.vmin < vmax: raise ValueError('Colorbar vmax must exceed vmin')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    tag=a.tag or (a.input.stem + ('_max' if a.percentile==100 else '_p'+f'{a.percentile:g}'))
    fig_path=a.output_dir/(tag+'.png')
    csv_path=a.output_dir/(tag+'.csv')
    plot_Z_azimuth_height_summary_ny_final(
        plot_matrix,AZ,plot_h,processed['peak'],main_bounds,seg_bounds,
        savepath=str(fig_path),vmin=a.vmin,vmax=vmax,dpi=a.dpi,
        y_lim=tuple(a.ylim),Z_label=('Max' if a.percentile==100 else int(a.percentile) if a.percentile.is_integer() else f'{a.percentile:g}'),
        BB_label=('Max' if a.percentile==100 else int(a.percentile) if a.percentile.is_integer() else f'{a.percentile:g}'))
    # Independent component stratiformity checks on final plotted detections.
    components={
        'BBB': (processed['lower'],main_ok, [r['lower'] for r in raw]),
        'BBP': (processed['peak'],(main_ok|seg_ok),[r['peak'] for r in raw]),
        'BBT': (processed['upper'],main_ok,[r['upper'] for r in raw]),
    }
    stats={}
    for name,(values,valid,_) in components.items():
        final=np.where(valid,values,np.nan)
        count=int(np.isfinite(final).sum())
        mean=np.nanmean(final) if count else np.nan
        cov=float(np.nanstd(final)/mean) if count and mean>0 else np.nan
        stats[name]=(count,cov,int(count>=a.stratiform_min_radials and np.isfinite(cov) and cov<=a.stratiform_max_cov))
    cols=['datetime_utc','radar','azimuth_deg','bb_characteristic','raw_height_m_asl',
          'smoothed_height_m_asl','raw_all_segment_bounds_m_asl','selected_segment_index',
          'smoothed_segment_lower_m_asl','smoothed_segment_upper_m_asl',
          'segment_fallback','outlier_01',
          'stratiform_01','stratiform_n_valid','stratiform_cov','status',
          'n_scans','input_file','reflectivity_percentile']
    m=re.search(r'(\d{8}T\d{6})Z',a.input.name)
    stamp=datetime.strptime(m.group(1),'%Y%m%dT%H%M%S').strftime('%Y-%m-%dT%H:%M:%SZ') if m else ''
    radar=a.input.name.split('_')[0]
    def clean(x):
        if x is None:return ''
        if isinstance(x,(float,np.floating)) and not np.isfinite(x):return ''
        return x
    with csv_path.open('w',newline='',encoding='utf-8') as fh:
        writer=csv.DictWriter(fh,fieldnames=cols)
        writer.writeheader()
        # Sort by timestamp, regular azimuth, and BBB/BBP/BBT for each azimuth.
        for i,r in enumerate(raw):
            for cname,(values,valid,rawvals) in components.items():
                count,cov,strat=stats[cname]
                mode='main' if main_ok[i] else ('segment_fallback' if cname=='BBP' and seg_ok[i] else '')
                accepted=bool(valid[i])
                row=dict(datetime_utc=stamp,radar=radar,azimuth_deg=AZ[i],bb_characteristic=cname,
                    raw_height_m_asl=rawvals[i],
                    smoothed_height_m_asl=values[i] if accepted else np.nan,
                    segment_fallback=int(mode == 'segment_fallback') if accepted else None,
                    outlier_01=int(processed['outlier'][i]),
                    stratiform_01=strat,stratiform_n_valid=count,stratiform_cov=cov,
                    status=('accepted_'+mode if accepted else ('outlier_unaccepted' if processed['outlier'][i] else ('candidate_unaccepted' if r['status'] == 'candidate' else r['status']))),
                    n_scans=n_scans,input_file=a.input.name,reflectivity_percentile=a.percentile)
                # Segment diagnostics are recorded once per azimuth, on the BBP row.
                if cname == 'BBP':
                    row.update(
                        raw_all_segment_bounds_m_asl=json.dumps(r['segments']),
                        selected_segment_index=r['segment_index'],
                        smoothed_segment_lower_m_asl=processed['seg_lower'][i],
                        smoothed_segment_upper_m_asl=processed['seg_upper'][i])
                writer.writerow({k:clean(v) for k,v in row.items()})
    print(f'Plot height: {a.ylim[0]:g}–{a.ylim[1]:g} m a.s.l.; reflectivity grid through {plot_h[-1]:g} m; colorbar: {a.vmin:g}–{vmax:.6g} linear Z')
    print('Figure:',fig_path)
    print('CSV:',csv_path)
    for name, (count,cov,flag) in stats.items():
        print(f'{name}: {count}/360 accepted; CV={cov:.4f}; stratiform_01={flag}')

def main():
    a = parse_args()
    if a.both:
        from copy import copy
        base = a.tag or a.input.stem
        # Accept --tag fig5 or --tag fig5a, consistently producing fig5a/fig5b.
        if a.tag and base.endswith(('a', 'b')) and len(base) > 1 and base[-2].isdigit():
            base = base[:-1]
        panels=[]
        for suffix, percentile in (('a', 100.0), ('b', 90.0)):
            item = copy(a)
            item.percentile = percentile
            item.tag = f'{base}{suffix}_{a.propagation}'
            print(f'Preparing {item.tag} ({percentile:g}th percentile)...')
            panels.append((item, prepare_one(item)))
        # Use the higher of the two data-derived 99th percentiles for a shared
        # linear color scale. Values above the limit will be saturated.
        common_max = a.vmax if a.vmax is not None else max(x[-1] for _, x in panels)
        print(f'Shared linear-Z colorbar: {a.vmin:g}–{common_max:.6g} mm^6 m^-3')
        for item, prepared in panels:
            print(f'\nGenerating {item.tag}...')
            run_one(item, prepared=prepared, shared_vmax=common_max)
    else:
        run_one(a)

if __name__ == '__main__':
    main()
