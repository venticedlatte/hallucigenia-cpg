"""Biological comparison figure for B&B paper.
   Carmen Mitchell

   Overlays HALLUCIGENIA simulation data against:
   1. Panarthropod phase offset scaling law (Nirody et al. 2021 ICB)
   2. Peripatus speed-frequency data (Manton 1950)
   3. Literature data points for other multi-legged organisms

   Ref: Nirody et al. 2021 Integr Comp Biol 61(2):710-720
        Manton 1950 J Linnean Soc 41:529-570
        Stegmann & Nirody 2023 J Exp Biol 226(Suppl 1):jeb245111
"""

import json
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from paper_style import (apply_style, panel_label, save_fig, COL2,
                         VCFG, GREEN, CYAN, BLUE, ORANGE, PINK)
apply_style()

RESULTS_DIR = Path(__file__).parent / 'results'

# load simulation analysis data
sim_data = {}
for vkey, fname in [('S', 'hallu-s_analysis.json'),
                     ('F', 'hallu-f_analysis.json'),
                     ('C', 'hallu-c_analysis.json')]:
    fpath = RESULTS_DIR / fname
    if fpath.exists():
        sim_data[vkey] = json.loads(fpath.read_text())

# load frequency sweep data
sweep_data = {}
for vkey, fname in [('S', 'hallu-s_freq_sweep.json'),
                     ('F', 'hallu-f_freq_sweep.json'),
                     ('C', 'hallu-c_freq_sweep.json')]:
    fpath = RESULTS_DIR / fname
    if fpath.exists():
        sweep_data[vkey] = json.loads(fpath.read_text())

# biological reference data

# Panarthropod phase offset scaling (Nirody et al. 2021)
bio_organisms = {
    'Drosophila':      {'n': 3,  'phi_low': 1/3,  'phi_high': 0.5,
                        'marker': '^', 'color': '#009E73'},
    'Tardigrade':      {'n': 4,  'phi_low': 0.25, 'phi_high': 0.5,
                        'marker': 'v', 'color': '#56B4E9'},
    'Stick insect':    {'n': 3,  'phi_low': 1/6,  'phi_high': 0.5,
                        'marker': '<', 'color': '#F0E442'},
    'Lithobius\n(centipede)': {'n': 15, 'phi_low': 1/15, 'phi_high': 0.25,
                        'marker': '>', 'color': '#CC79A7'},
    'Peripatopsis':    {'n': 19, 'phi_low': 1/19, 'phi_high': 0.15,
                        'marker': 'P', 'color': '#D55E00'},
}

# Manton 1950 Peripatus speed data
bio_freq  = [0.5, 0.8, 1.2, 1.6, 2.0]
bio_speed = [5.0, 8.0, 13.0, 18.0, 25.0]  # mm/s


# Figure: 3-panel biological comparison
fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.6))

# (a) Phase offset scaling: phi vs number of valve groups
ax = axes[0]
panel_label(ax, 'a')

# theoretical 1/n curve (Nirody 2021)
n_range = np.linspace(2, 30, 200)
phi_1n = 1.0 / n_range
ax.plot(n_range, phi_1n, 'k--', linewidth=1.2, alpha=0.5,
        label=r'$\phi_I = 1/n$ (Nirody 2021)')

# half-wavelength curve (our CPG design)
phi_half = 0.5 / n_range
ax.plot(n_range, phi_half, 'k:', linewidth=0.8, alpha=0.35,
        label=r'$\phi_I = 1/2n$ (half-wave CPG)')

# biological data points
for name, bio in bio_organisms.items():
    ax.plot([bio['n'], bio['n']], [bio['phi_low'], bio['phi_high']],
            '-', color=bio['color'], linewidth=1.8, alpha=0.5)
    ax.plot(bio['n'], bio['phi_low'], bio['marker'], color=bio['color'],
            markersize=6, markeredgecolor='black', markeredgewidth=0.4,
            label=name, zorder=5)

# our simulation data
for vkey in ['S', 'F', 'C']:
    if vkey not in sim_data:
        continue
    d = sim_data[vkey]
    cfg = VCFG[vkey]
    measured_lag_deg = abs(d['phase_lag']['mean_lag_deg'])
    phi_measured = measured_lag_deg / 360.0
    n_plot = cfg['n_valves']

    ax.plot(n_plot, phi_measured, cfg['marker'], color=cfg['color'],
            markersize=8, markeredgecolor='black', markeredgewidth=0.8,
            label=cfg['label'], zorder=10)

ax.set_xlabel('Number of Ipsilateral Leg Groups')
ax.set_ylabel(r'Phase Offset $\phi_I$ (fraction of stride cycle)')
ax.set_xlim(1, 30)
ax.set_ylim(0, 0.55)
ax.legend(fontsize=5.5, loc='upper right', ncol=1)


# (b) Speed vs frequency: simulation vs Peripatus
ax = axes[1]
panel_label(ax, 'b')

ax.plot(bio_freq, bio_speed, '^--', color=GREEN, markersize=6,
        markeredgecolor='black', markeredgewidth=0.4,
        label='Peripatus (Manton 1950)', zorder=5)

for vkey in ['S', 'F', 'C']:
    if vkey not in sweep_data:
        continue
    cfg = VCFG[vkey]
    data = sweep_data[vkey]
    freqs = [d['freq_hz'] for d in data]
    speeds = [d['mean_speed_mm_s'] for d in data]
    ax.plot(freqs, speeds, f"{cfg['marker']}-", color=cfg['color'],
            label=f"{cfg['name']} (sim)")

ax.set_xlabel('CPG / Stride Frequency (Hz)')
ax.set_ylabel('Speed (mm/s)')
ax.legend()


# (c) Normalized speed: body lengths per second
ax = axes[2]
panel_label(ax, 'c')

for vkey in ['S', 'F', 'C']:
    if vkey not in sweep_data or vkey not in sim_data:
        continue
    cfg = VCFG[vkey]
    data = sweep_data[vkey]
    body_len_mm = sim_data[vkey]['energy']['body_length_mm']

    freqs = [d['freq_hz'] for d in data]
    bl_per_s = [d['mean_speed_mm_s'] / body_len_mm for d in data]

    ax.plot(freqs, bl_per_s, f"{cfg['marker']}-", color=cfg['color'],
            label=f"{cfg['name']} ({body_len_mm}mm)")

# Peripatus: ~50mm body length (Manton's specimens)
peripatus_bl = 50.0
bio_bl_per_s = [s / peripatus_bl for s in bio_speed]
ax.plot(bio_freq, bio_bl_per_s, '^--', color=GREEN, markersize=6,
        markeredgecolor='black', markeredgewidth=0.4,
        label=f'Peripatus (~{peripatus_bl:.0f}mm)')

ax.set_xlabel('CPG / Stride Frequency (Hz)')
ax.set_ylabel('Speed (body lengths / s)')
ax.legend()


plt.tight_layout()
save_fig(fig, RESULTS_DIR, 'bio_comparison')
print("Done.")
