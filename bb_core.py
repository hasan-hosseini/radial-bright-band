#!/usr/bin/env python3
"""
Core functions for radial bright-band (BB) detection.

Implements reflectivity gridding, raw BB characteristic
detection, rise-fall segmentation, outlier filtering,
and azimuthal smoothing.

Author: Hasan Hosseini
Affiliation: Swedish Meteorological and Hydrological Institute (SMHI)
Reference: Hosseini et al. (2026)
"""

from __future__ import annotations
import re
import numpy as np
import h5py

def ischange_gradient(array1D, max_changes=6, grad_percentile=80, max_percentile=99):
    """
    Identifies change segments in a 1D array based on significant positive and negative gradients.

    Each segment:
    - Starts at the first significant positive gradient.
    - Ends at the last significant negative gradient before the next significant positive gradient or end of array.
    - Segments with unresolved descending gradients at the profile end are discarded.

    The function adapts gradient thresholds using percentile values and limits the number of detected segments.

    Parameters:
        array1D (np.ndarray): Input 1D array (e.g., reflectivity profile). Must not contain NaNs.
        max_changes (int): Maximum number of change points (i.e., 2 * segments) allowed.
        grad_percentile (int): Starting percentile to define gradient threshold.
        max_percentile (int): Maximum percentile to test if too many change points are found.

    Returns:
        starts (List[int]): Indices where segments begin (start of significant rise).
        ends (List[int]): Indices where segments end (end of significant fall).
        S1 (np.ndarray): Array of same length as input; contains segment-wise mean.
        S2 (np.ndarray): Segment-wise variance.
        S3 (np.ndarray): Segment-wise maximum value.
        S4 (np.ndarray): Index (absolute, not relative) of maximum value within each segment.
    """
    array1D = np.asarray(array1D, dtype=float)

    if np.isnan(array1D).any():
        raise ValueError("Input contains NaNs. Remove or interpolate them first.")

    n = len(array1D)
    if n < 3:
        return [], [], np.zeros(n, dtype=bool), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)

    grad = np.diff(array1D)

    def alternating_first_last_gradients(grad, abspos_threshold, absneg_threshold):
        """
        Detects change points by pairing:
        - Each segment starts with the first significant +gradient.
        - Ends at the last significant -gradient before next +gradient or n.
        """
        n = len(grad)
        i = 1
        starts = []
        ends = []

        while i < n:
            # Step 1: First significant positive gradient
            pos = next((j for j in range(i, n) if grad[j] >= abspos_threshold), None)
            if pos is None:
                break
            i = pos + 1
            starts.append(i)

            # Step 2: First significant negative after the positive
            neg = next((j for j in range(i, n) if grad[j] <= -absneg_threshold), None)
            if neg is None:
                break

            # Step 3: Next significant positive or n
            next_pos_OR_n = next((j for j in range(neg + 1, n) if grad[j] >= abspos_threshold), n)

            # Step 4: Last significant negative before next positive (or end)
            negs_in_range = [j for j in range(neg, next_pos_OR_n) if grad[j] <= -absneg_threshold]
            last_neg = negs_in_range[-1]

            # Step 5:
            if next_pos_OR_n == n and len(negs_in_range) == 1 and last_neg+1 == n:
                break

            i = last_neg + 1
            ends.append(i)

        return np.array(starts), np.array(ends)

    # Compute adaptive thresholds
    pos_grads = grad[grad >= 0]
    neg_grads = -grad[grad <= 0]  # take absolute values

    percentile = grad_percentile
    while percentile <= max_percentile:
        pos_thresh = np.percentile(pos_grads, percentile) if len(pos_grads) > 0 else 0
        neg_thresh = np.percentile(neg_grads, percentile) if len(neg_grads) > 0 else 0

        min_thresh = abs(array1D[0]) * 0.06 # note that array1D[0] >= THRESH_Z
        pos_thresh = max(pos_thresh, min_thresh)
        neg_thresh = max(neg_thresh, min_thresh)

        starts, ends = alternating_first_last_gradients(grad, pos_thresh, neg_thresh)
        if len(starts) <= max_changes/2:
            break
        percentile += 1

    # === Step 3: Create outputs
    S1 = np.full(n, np.nan)
    S2 = np.full(n, np.nan)
    S3 = np.full(n, np.nan)
    S4 = np.full(n, np.nan)

    # Ensure starts and ends are aligned
    if len(ends) < len(starts):
        starts = starts[:len(ends)]

    # New list of segment boundaries
    segment_boundaries = []

    # Handle segment before the first start
    if len(starts) > 0 and starts[0] > 0:
        segment_boundaries.append((0, starts[0]))

    # Handle segments between end[i] and start[i+1]
    for i in range(len(ends) - 1):
        segment_boundaries.append((starts[i], ends[i]))
        if ends[i] < starts[i+1]:
            segment_boundaries.append((ends[i], starts[i+1]))

    # Handle the last matched segment
    if len(ends) > 0:
        segment_boundaries.append((starts[-1], ends[-1]))

        # Handle segment after the last end
        if ends[-1] < n:
            segment_boundaries.append((ends[-1], n))
    elif len(starts) == 0:
        # If no segments found, entire array is one segment
        segment_boundaries.append((0, n))

    # Compute S1–S4 for all segments
    for s, e in segment_boundaries:
        segment = array1D[s:e]
        S1[s:e] = np.mean(segment)
        S2[s:e] = np.var(segment)
        S3[s:e] = np.max(segment)
        S4[s:e] = s + np.argmax(segment)

    starts = np.asarray(starts).astype(int)
    ends = np.asarray(ends).astype(int)
    S4 = np.asarray(S4).astype(int)

    return starts, ends, S1, S2, S3, S4

def smooth_heights_triangular(common_azimuth, heights, sector=60):
    """
    Smooth heights over azimuths using triangular weights within a defined sector.
    
    Args:
        common_azimuth (np.ndarray): Azimuth array (shape: [N], e.g., 0 to 359).
        heights (np.ndarray): Height values to smooth (same length as common_azimuth).
        sector (int): Sector size in degrees (must be even, e.g., 60).

    Returns:
        np.ndarray: Smoothed height values (same shape as input).
    """

    def triangular_weight(d, half_sector):
        return max(0, 1 - abs(d) / half_sector)

    smooth_values = []
    half_sector = sector // 2
    az_len = len(common_azimuth)
    heights = np.asarray(heights)

    for i in range(az_len):
        indices = [(i + d) % az_len for d in range(-half_sector, half_sector + 1)]
        weights = np.array([triangular_weight(d, half_sector) for d in range(-half_sector, half_sector + 1)])
        values = heights[indices]

        valid = ~np.isnan(values)
        if np.any(valid):
            w = weights[valid]
            if w.sum() == 0:
                smooth_values.append(np.nan)
            else:
                smooth_values.append(np.average(values[valid], weights=w))
        else:
            smooth_values.append(np.nan)

    return np.array(smooth_values)

def attrs(group, key, default=None):
    value = group.attrs.get(key, default)
    if isinstance(value, bytes): return value.decode()
    if isinstance(value, np.ndarray) and value.size == 1: return value.item()
    return value

def numbered_keys(group, prefix):
    return sorted((name for name in group if re.fullmatch(prefix + r"\d+", name)),
                  key=lambda name: int(name[len(prefix):]))

R_E = 6371000.0

def profile_grid(path, height_grid, percentile=100, propagation='straight', quantity='DBZH'):
    """Map native radar gates to nearest common azimuth and MRR-aligned height bins.

    Reproduces original startazA + 0.5 and range = i*rscale conventions.
    Uses reflectivity only.
    """
    nheight=len(height_grid)
    buckets=[[[] for _ in range(nheight)] for _ in range(360)]
    with h5py.File(path,'r') as f:
        radar_alt=float(attrs(f['where'],'height'))
        dkeys=numbered_keys(f,'dataset')
        for dkey in dkeys:
            ds=f[dkey]; el=np.deg2rad(float(attrs(ds['where'],'elangle')))
            data_group=None
            for key in numbered_keys(ds,'data'):
                group=ds[key]
                if 'what' in group and str(attrs(group['what'],'quantity','')).upper()==quantity.upper():
                    data_group=group;break
            if data_group is None: raise ValueError(f'No {quantity} in {dkey}')
            w=data_group['what']; encoded=data_group['data'][()]
            mask=encoded==attrs(w,'nodata',-99999)
            encoded=encoded.astype(float)
            undetect=attrs(w,'undetect',None)
            if undetect is not None:
                encoded[encoded==undetect]=( -40.0-float(attrs(w,'offset',0)))/float(attrs(w,'gain',1))
            dbz=encoded*float(attrs(w,'gain',1))+float(attrs(w,'offset',0))
            dbz[mask]=np.nan
            Z=10**(dbz/10)
            nrays,nbins=Z.shape
            az=np.mod(np.asarray(attrs(ds['how'],'startazA'),float)+0.5,360)
            if len(az)!=nrays: raise ValueError(f'Incorrect azimuth count in {dkey}')
            # Original law-of-cosines method; standard uses parabolic 4/3-radius approximation.
            rng=np.arange(nbins)*float(attrs(ds['where'],'rscale'))
            if propagation=='straight':
                H0=R_E+radar_alt
                h=np.sqrt(H0**2+rng**2+2*H0*rng*np.sin(el))-R_E
            else:
                h=radar_alt+rng*np.sin(el)+rng**2/(2*(4/3)*R_E)
            use=(h>=height_grid[0]-0.49*(height_grid[1]-height_grid[0])) & (h<=height_grid[-1]+0.49*(height_grid[1]-height_grid[0]))
            js=np.where(use)[0]
            if not len(js):continue
            hidx=np.argmin(abs(height_grid[:,None]-h[js]),axis=0)
            aidx=np.argmin(abs((np.arange(360)+0.5)[:,None]-az[None,:]),axis=0)
            for i in range(nrays):
                for j,k in zip(js,hidx):
                    v=Z[i,j]
                    if np.isfinite(v):buckets[aidx[i]][k].append(v)
    matrix=np.full((360,nheight),np.nan)
    for i in range(360):
        for j in range(nheight):
            if buckets[i][j]:matrix[i,j]=np.percentile(buckets[i][j],percentile)
        # Original interpolation of only isolated missing heights; avoid wrap on vertical endpoints.
        for j in range(1,nheight-1):
            if np.isnan(matrix[i,j]) and np.isfinite(matrix[i,j-1]) and np.isfinite(matrix[i,j+1]):
                matrix[i,j]=(matrix[i,j-1]+matrix[i,j+1])/2
    return matrix, len(dkeys)

def raw_components(profile,h,threshold=1.0,min_bins=5,gradient_percentile=85,max_changes=6):
    result={'peak':np.nan,'lower':np.nan,'upper':np.nan,'segment':(np.nan,np.nan),
            'segments':[],'segment_index':None,'status':'invalid_profile','n_valid':int(np.isfinite(profile).sum())}
    if not np.isfinite(profile[0]) or profile[0] < threshold:
        result['status']='lowest_level_below_threshold';return result
    if np.sum(np.isfinite(profile)&(profile>threshold))<min_bins:
        result['status']='too_few_wet_levels';return result
    # Original extract_and_smooth_heights rejects ANY remaining NaN.
    if np.isnan(profile).any():
        result['status']='missing_vertical_levels';return result
    peak_idx=int(np.argmax(profile)); peak=h[peak_idx]
    result['peak']=float(peak)
    grad=np.insert(np.diff(profile),0,np.nan)
    pos=int(np.nanargmax(grad)); neg=int(np.nanargmin(grad))
    lower=h[pos-1] if pos>0 else h[0]
    upper=h[neg]
    if lower<=peak<=upper:
        result['lower'],result['upper']=float(lower),float(upper)
    starts,ends,*_=ischange_gradient(profile,max_changes=max_changes,grad_percentile=gradient_percentile)
    segments=[(float(h[s-1] if s>0 else h[0]),float(h[e])) for s,e in zip(starts,ends)]
    result['segments']=segments
    # Original fallback chooses narrowest peak-containing segment, preferring
    # segments spanning main gradient-extrema bounds when they exist.
    candidates=[(index,a,b) for index,(a,b) in enumerate(segments) if a<=peak<=b]
    if candidates:
        valid_main=np.isfinite(result['lower']) and np.isfinite(result['upper'])
        spanning=[c for c in candidates if c[1]<=result['lower'] and c[2]>=result['upper']] if valid_main else []
        index,a,b=min(spanning or candidates,key=lambda c:c[2]-c[1])
        result['segment']=(a,b);result['segment_index']=int(index)
    result['status']='candidate';return result

def mask_and_smooth(raw,sector=6,zlimit=2.5):
    """Remove height outliers and apply circular triangular smoothing."""
    n=len(raw)
    peaks=np.array([r['peak'] for r in raw],float)
    lower=np.array([r['lower'] for r in raw],float)
    upper=np.array([r['upper'] for r in raw],float)
    seglo=np.array([r['segment'][0] for r in raw],float)
    seghi=np.array([r['segment'][1] for r in raw],float)
    def outlier(v):
        if np.isfinite(v).sum()<2:return np.zeros(n,bool)
        sd=np.nanstd(v)
        return np.isfinite(v)&(abs(v-np.nanmean(v))/sd>zlimit) if sd>0 else np.zeros(n,bool)
    flagged=outlier(peaks)|outlier(lower)|outlier(upper)
    for v in (peaks,lower,upper,seglo,seghi):v[flagged]=np.nan
    az=np.arange(360)+0.5
    sm=[smooth_heights_triangular(az,v,sector=sector) for v in (peaks,lower,upper,seglo,seghi)]
    sp,sl,su,sslo,sshi=sm
    main_valid=np.isfinite(sp)&np.isfinite(sl)&np.isfinite(su)&(sl<=sp)&(sp<=su)
    seg_valid=np.isfinite(sp)&np.isfinite(sslo)&np.isfinite(sshi)&(sslo<=sp)&(sp<=sshi)
    accepted=main_valid|seg_valid
    return dict(raw_peak=np.array([r['peak'] for r in raw]),peak=sp,lower=sl,upper=su,
                seg_lower=sslo,seg_upper=sshi,outlier=flagged,
                accepted=accepted,main_valid=main_valid,seg_valid=seg_valid)
