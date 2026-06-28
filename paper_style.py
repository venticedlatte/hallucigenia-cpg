"""Publication figure style for B&B / OCEANS submissions.
   Carmen Mitchell

   Colorblind-safe palette (Wong 2011, Nature Methods 8:441).
   Sized for Bioinspiration & Biomimetics column widths.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# B&B / IOP column widths (inches)
COL1 = 3.5    # single column ~89mm
COL2 = 7.2    # double column ~183mm

# colorblind-safe palette (Wong 2011)
BLUE    = '#0072B2'
ORANGE  = '#E69F00'
PINK    = '#CC79A7'
GREEN   = '#009E73'
RED     = '#D55E00'
CYAN    = '#56B4E9'
YELLOW  = '#F0E442'
BLACK   = '#000000'

# variant config (single source of truth for all plots)
VCFG = {
    'S': {'name': 'HALLU-S', 'n_pairs': 7,  'n_valves': 7,
          'color': BLUE,   'marker': 'o', 'label': 'HALLU-S (7p)'},
    'F': {'name': 'HALLU-F', 'n_pairs': 9,  'n_valves': 9,
          'color': ORANGE, 'marker': 's', 'label': 'HALLU-F (9p)'},
    'C': {'name': 'HALLU-C', 'n_pairs': 25, 'n_valves': 8,
          'color': PINK,   'marker': 'D', 'label': 'HALLU-C (25p)'},
}

BIO_COLOR = GREEN    # biological reference data
DIRECT_COLOR = BLUE
RETRO_COLOR = RED

DPI = 300
FORMATS = ['png', 'pdf']

# gait diagram colormap
GAIT_CMAP = 'cividis'   # colorblind-safe sequential


def apply_style():
    """Set rcParams for publication figures."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'axes.linewidth': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.15,
        'grid.linewidth': 0.4,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'lines.linewidth': 1.4,
        'lines.markersize': 5,
        'errorbar.capsize': 2,
        'figure.dpi': 150,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })


def panel_label(ax, label, x=-0.12, y=1.08):
    """Add bold panel label (a), (b), etc. in axes coordinates."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='left')


def save_fig(fig, outdir, name):
    """Save figure in all configured formats."""
    outdir = Path(outdir)
    for fmt in FORMATS:
        p = outdir / f'{name}.{fmt}'
        fig.savefig(p, dpi=DPI, bbox_inches='tight')
    print(f"  Saved: {outdir / name}.{{{','.join(FORMATS)}}}")
    plt.close(fig)
