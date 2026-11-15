"""
Plot azimuthal reflectivity profiles and detected bright-band (BB)
characteristics, as illustrated in Figures 4 and 5 of Hosseini et al. (2026).

Contains plot_Z_azimuth_height_summary_ny_final, which displays
linear reflectivity (Z), accepted smoothed BB peaks and boundaries,
and significant rise–fall segment boundaries.

Author: Hasan Hosseini
Affiliation: Swedish Meteorological and Hydrological Institute (SMHI)
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_Z_azimuth_height_summary_ny_final(
    ZMATRIX,
    common_azimuth,
    common_height,
    peak_heights,
    main_bounds,
    significant_bounds,
    savepath=None,
    vmax=None,
    vmin=0,
    figsize=(8, 4),
    layout="constrained",
    cmap_name="viridis",
    dpi=300,
    cbar_pad=0.001,
    y_lim=(300, 3000),
    ytick_step=500,
    grid=True,
    title=None,
    Z_label=None,
    BB_label=None,
):

    """
    Z-only summary (8x4): Azimuth on x, Height on y.
    Colormap always shown over y_lim, but overlays (peak/bounds/segment) are shown ONLY if
    the BB acceptance criteria are met per azimuth:

      Accept if:
        A) main bounds valid and (main_bot <= peak <= main_top), OR
        B) otherwise, significant segment bounds valid and (sig_bot <= peak <= sig_top)

    Additionally, overlays are hidden if they are not fully inside y_lim.
    This prevents plotting BB features outside your displayed height window.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import math
    from matplotlib.ticker import FuncFormatter

    # --- helper: bin edges from centers ---
    def edges_from_centers(x):
        x = np.asarray(x, dtype=float)
        if x.size < 2:
            dx = 1.0
            return np.array([x[0] - dx / 2, x[0] + dx / 2], dtype=float)
        edges = np.empty(x.size + 1, dtype=float)
        edges[1:-1] = (x[:-1] + x[1:]) / 2
        edges[0] = x[0] - (x[1] - x[0]) / 2
        edges[-1] = x[-1] + (x[-1] - x[-2]) / 2
        return edges

    az_edges = edges_from_centers(common_azimuth)
    h_edges = edges_from_centers(common_height)

    fig, ax = plt.subplots(figsize=figsize, layout=layout)

    # --- Colormap: always from ZMATRIX (full), y_lim only clips the view ---
    Z = np.array(ZMATRIX, dtype=float)
    Zm = np.ma.masked_invalid(Z)

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("white")

    mesh = ax.pcolormesh(
        az_edges,
        h_edges,
        Zm.T,  # height x azimuth
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    # --- Colorbar: integer-only ticks and ×10^n label ---
    cbar = fig.colorbar(mesh, ax=ax, pad=cbar_pad, shrink=0.85, aspect=20,
                        # extend="max" if vmax is not None else "neither"
                        )

    vmin_cb, vmax_cb = mesh.get_clim()
    z_ref = vmax_cb if np.isfinite(vmax_cb) and vmax_cb > 0 else np.nanmax(Z)
    if not np.isfinite(z_ref) or z_ref <= 0:
        scale_exp = 0
    else:
        scale_exp = max(0, int(math.ceil(math.log10(z_ref))) - 1)

    factor = 10 ** scale_exp

    # One-unit ticks in the scaled colorbar coordinates (0, 1, 2, ...).
    # Keep the true vmax: no forced rounding or additional saturation.
    lower_tick = int(np.ceil(vmin_cb / factor))
    upper_tick = int(np.floor(vmax_cb / factor))
    tick_values = np.arange(lower_tick, upper_tick + 1, dtype=float) * factor
    if tick_values.size:
        cbar.set_ticks(tick_values)
        cbar.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: f"{x / factor:g}")
        )
    cbar.ax.set_ylim(vmin_cb, vmax_cb)
    # ---- Colorbar label (robust mathtext; avoids broken underscores) ----
    def _ordinal_suffix(n: int) -> str:
        n = int(n)
        if 10 <= (n % 100) <= 20:
            return "th"
        return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")

    # ---- Colorbar label: italic Z, upright subscript + units ----
    if Z_label is None:
        Z_label_math = r"$Z$"
    else:
        if isinstance(Z_label, (int, np.integer)):
            suf = _ordinal_suffix(int(Z_label))
            # italic Z, upright subscript text
            Z_label_math = rf"$Z_{{\mathrm{{{int(Z_label)}{suf}}}}}$"
        else:
            zstr = str(Z_label)
            # italic Z, upright subscript text (e.g., Max)
            Z_label_math = rf"$Z_{{\mathrm{{{zstr}}}}}$"

    cbar.set_label(
        rf"{Z_label_math} "
        rf"$\mathrm{{(\times 10^{{{scale_exp}}}\,mm^6\,m^{{-3}})}}$"
    )

    # ---- Legend label: upright "BB" + true subscript, no italics ----
    if BB_label is None:
        BB_legend = r"$\mathrm{BB}$"
    else:
        # Integer percentile -> BB_{90th} with upright text
        if isinstance(BB_label, (int, np.integer)):
            suf = _ordinal_suffix(int(BB_label))
            BB_legend = rf"$\mathrm{{BB}}_{{\mathrm{{{int(BB_label)}{suf}}}}}$"
        else:
            # String tag like "Max" -> BB_{Max} with upright text
            bstr = str(BB_label)
            BB_legend = rf"$\mathrm{{BB}}_{{\mathrm{{{bstr}}}}}$"

    # --- Convert overlays to arrays ---
    peak = np.asarray(peak_heights, dtype=float)
    main_bot = np.array([b[0] for b in main_bounds], dtype=float)
    main_top = np.array([b[1] for b in main_bounds], dtype=float)
    sig_bot = np.array([b[0] for b in significant_bounds], dtype=float)
    sig_top = np.array([b[1] for b in significant_bounds], dtype=float)

    y0, y1 = float(y_lim[0]), float(y_lim[1])

    # --- Validity within plot window ---
    def inside_window(a, b):
        return np.isfinite(a) & np.isfinite(b) & (a >= y0) & (b <= y1) & (a <= b)

    main_inwin = inside_window(main_bot, main_top)
    sig_inwin  = inside_window(sig_bot, sig_top)
    peak_inwin = np.isfinite(peak) & (peak >= y0) & (peak <= y1)

    # --- Acceptance logic (per azimuth) ---
    accept_main = main_inwin & peak_inwin & (main_bot <= peak) & (peak <= main_top)
    accept_sig  = sig_inwin  & peak_inwin & (sig_bot  <= peak) & (peak <= sig_top)

    # Peak is accepted if either condition holds (consistent with CSV mask)
    accept = accept_main | accept_sig

    # Mask peak where not accepted
    peak_plot = peak.copy()
    peak_plot[~accept] = np.nan

    # Show main bounds where main is valid (independent of sig)
    main_bot_plot = main_bot.copy()
    main_top_plot = main_top.copy()
    main_bot_plot[~accept_main] = np.nan
    main_top_plot[~accept_main] = np.nan

    # Show sig segments where sig is valid (independent of main)
    sig_bot_plot = sig_bot.copy()
    sig_top_plot = sig_top.copy()
    sig_bot_plot[~accept_sig] = np.nan
    sig_top_plot[~accept_sig] = np.nan

    # --- Plot overlays (consistent styling) ---
    lw_main = 1.5
    lw_sig = 2.0

    # Bounds (red) and sig segments (gray)
    # ax.plot(common_azimuth, main_bot_plot, "r-", lw=lw_main, label=rf"{BB_legend} bounds")
    # ax.plot(common_azimuth, main_top_plot, "r-", lw=lw_main, label="_nolegend_")
    # ax.plot(common_azimuth, peak_plot, "r--", lw=lw_main, label=rf"{BB_legend} peak")
    ax.plot(common_azimuth, main_bot_plot, "r-", lw=lw_main, label="BB bounds")
    ax.plot(common_azimuth, main_top_plot, "r-", lw=lw_main, label="_nolegend_")
    ax.plot(common_azimuth, peak_plot, "r--", lw=lw_main, label="BB peak")

    ax.plot(
        common_azimuth, sig_bot_plot,
        color="#6D6D6D", linestyle=":", lw=lw_sig,
        label="Significant rise–fall segments"
    )

    ax.plot(common_azimuth, sig_top_plot, color="#6D6D6D", linestyle=":", lw=lw_sig, label="_nolegend_")

    # --- Axes formatting ---
    ax.set_xlabel("Azimuth (°)")
    ax.set_ylabel("Height (m a.s.l.)")
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 30))

    ax.set_ylim(y0, y1)
    ax.set_yticks(np.arange(ytick_step, y1 + 1, ytick_step))
    if title:
        ax.set_title(title)

    ax.legend(loc="upper center", ncol=3, frameon=True)

    if grid:
        ax.grid(True, which="major", axis="both", linestyle=":", lw=0.25, color="lightgray")

    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
    else:
        plt.show()
