"""Generate multi-variant comparison plot from sweep data.
   Carmen Mitchell"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from paper_style import (apply_style, panel_label, save_fig, COL2,
                         VCFG as PVCFG, GREEN, BLUE, ORANGE)
apply_style()

results_dir = Path(__file__).parent / 'results'

variants = {
    'S': {'file': 'freq_sweep.json', 'name': 'HALLU-S', 'pairs': 7},
    'F': {'file': 'hallu-f_freq_sweep.json', 'name': 'HALLU-F', 'pairs': 9},
    'C': {'file': 'hallu-c_freq_sweep.json', 'name': 'HALLU-C', 'pairs': '25/8z'},
}

# biological data (Manton 1950)
bio_freq  = [0.5, 0.8, 1.2, 1.6, 2.0]
bio_speed = [5.0, 8.0, 13.0, 18.0, 25.0]  # mm/s, Peripatus

fig, axes = plt.subplots(2, 2, figsize=(COL2, COL2 * 0.75))

for vkey, vcfg in variants.items():
    fpath = results_dir / vcfg['file']
    if not fpath.exists():
        print(f"  {vcfg['name']}: file not found, skipping")
        continue

    pcfg = PVCFG[vkey]
    data = json.loads(fpath.read_text())
    freqs = [d['freq_hz'] for d in data]
    speeds = [d['mean_speed_mm_s'] for d in data]
    step_lens = [d['step_length_mm'] for d in data]
    straightness = [d['straightness'] for d in data]
    cots = [d['cost_of_transport'] for d in data]

    axes[0, 0].plot(freqs, speeds, f"{pcfg['marker']}-", color=pcfg['color'],
                    label=pcfg['label'])
    axes[0, 1].plot(freqs, step_lens, f"{pcfg['marker']}-", color=pcfg['color'],
                    label=pcfg['label'])
    axes[1, 0].plot(freqs, straightness, f"{pcfg['marker']}-", color=pcfg['color'],
                    label=pcfg['label'])
    axes[1, 1].plot(freqs, cots, f"{pcfg['marker']}-", color=pcfg['color'],
                    label=pcfg['label'])

# bio reference
axes[0, 0].plot(bio_freq, bio_speed, '^--', color=GREEN,
                alpha=0.7, label='Peripatus (Manton 1950)')

panel_label(axes[0, 0], 'a')
axes[0, 0].set_xlabel('CPG Frequency (Hz)')
axes[0, 0].set_ylabel('Mean Speed (mm/s)')
axes[0, 0].legend()

panel_label(axes[0, 1], 'b')
axes[0, 1].axhline(15.0, color='gray', linestyle='--', linewidth=0.6, alpha=0.5)
axes[0, 1].set_xlabel('CPG Frequency (Hz)')
axes[0, 1].set_ylabel('Step Length (mm)')
axes[0, 1].legend()

panel_label(axes[1, 0], 'c')
axes[1, 0].set_xlabel('CPG Frequency (Hz)')
axes[1, 0].set_ylabel('Straightness (0\u20131)')
axes[1, 0].set_ylim(0, 1.05)
axes[1, 0].legend()

panel_label(axes[1, 1], 'd')
axes[1, 1].set_xlabel('CPG Frequency (Hz)')
axes[1, 1].set_ylabel('Cost of Transport (a.u.)')
axes[1, 1].legend()

plt.tight_layout()
save_fig(fig, results_dir, 'variant_comparison')

# also generate the coupling sweep figure
coupling_file = results_dir / 'hallu-s_coupling_sweep.json'
if coupling_file.exists():
    cdata = json.loads(coupling_file.read_text())
    Ks = [d['K'] for d in cdata]
    speeds = [d['mean_speed_mm_s'] for d in cdata]
    step_lens = [d['step_length_mm'] for d in cdata]
    straightness = [d['straightness'] for d in cdata]

    fig2, axes2 = plt.subplots(1, 3, figsize=(COL2, 2.4))

    panel_label(axes2[0], 'a')
    axes2[0].plot(Ks, speeds, 'o-', color=BLUE)
    axes2[0].set_xlabel('Coupling Gain $K$')
    axes2[0].set_ylabel('Speed (mm/s)')
    axes2[0].axvline(2.0, color='#D55E00', linestyle='--', linewidth=0.8, alpha=0.6,
                     label='Design $K$=2.0')
    axes2[0].legend()

    panel_label(axes2[1], 'b')
    axes2[1].plot(Ks, step_lens, 'o-', color=ORANGE)
    axes2[1].axhline(15.0, color='gray', linestyle='--', linewidth=0.6, alpha=0.5)
    axes2[1].set_xlabel('Coupling Gain $K$')
    axes2[1].set_ylabel('Step Length (mm)')

    panel_label(axes2[2], 'c')
    axes2[2].plot(Ks, straightness, 'o-', color='#333333')
    axes2[2].set_xlabel('Coupling Gain $K$')
    axes2[2].set_ylabel('Straightness (0\u20131)')
    axes2[2].set_ylim(0, 1.05)
    axes2[2].axvline(2.0, color='#D55E00', linestyle='--', linewidth=0.8, alpha=0.6)

    plt.tight_layout()
    save_fig(fig2, results_dir, 'coupling_analysis')

print("\nDone.")
