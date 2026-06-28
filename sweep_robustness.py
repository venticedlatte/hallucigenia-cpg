"""HALLUCIGENIA Robustness Analysis
   Carmen Mitchell

   Three systematic studies for the B&B paper:
   1. Direct vs retrograde metachronal wave comparison
   2. Phase noise robustness (Kuramoto basin of attraction)
   3. Incline performance across body morphologies

   Ref: Kuramoto 1984 (Chemical Oscillations, Waves, and Turbulence)
        Manton 1950 J Linnean Soc 41:529-570 (Peripatus wave direction)
        Oliveira 2012 PLoS ONE 7(12):e51220 (Onychophora phase lag)

   Usage:
       python sweep_robustness.py --all
       python sweep_robustness.py --wave-direction
       python sweep_robustness.py --noise
       python sweep_robustness.py --slope
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import mujoco

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sim_cpg import (VARIANTS, COUPLING_GAIN, VALVE_DUTY,
                     KuramotoCPG, load_model, get_actuator_mapping,
                     run_simulation, compute_metrics)

from paper_style import (apply_style, panel_label, save_fig, COL2,
                         VCFG as PVCFG, DIRECT_COLOR, RETRO_COLOR)
apply_style()

RESULTS_DIR = Path(__file__).parent / 'results'
MODELS_DIR  = Path(__file__).parent / 'models'
DURATION    = 20.0


def run_single(variant_key, freq=0.8, phase_offset=None, coupling=COUPLING_GAIN,
               duty=VALVE_DUTY, phase_noise=0.0, slope_deg=0.0,
               return_results=False):
    """Run one simulation and return metrics + mean coherence."""
    vcfg = VARIANTS[variant_key]
    model_path = MODELS_DIR / vcfg['model']
    model, data = load_model(str(model_path))
    actuator_map = get_actuator_mapping(model, variant_key)

    if phase_offset is None:
        phase_offset = vcfg['phase_offset']

    # apply slope by tilting gravity
    if slope_deg != 0.0:
        angle_rad = np.radians(slope_deg)
        g = 9.81
        model.opt.gravity[0] = -g * np.sin(angle_rad)
        model.opt.gravity[2] = -g * np.cos(angle_rad)

    cpg = KuramotoCPG(vcfg['n_valves'], freq, phase_offset, coupling, duty,
                      phase_noise=phase_noise)
    results = run_simulation(model, data, cpg, DURATION, actuator_map,
                             variant_key=variant_key)
    metrics = compute_metrics(results)

    # steady-state coherence (after 3s)
    t = results['t']
    mask = t > 3.0
    mean_coherence = float(np.mean(results['coherence'][mask]))

    if return_results:
        return metrics, mean_coherence, results
    return metrics, mean_coherence


# 1. Wave direction comparison

def wave_direction_comparison(outdir):
    """Compare direct (anterior-to-posterior) vs retrograde (posterior-to-anterior)
    metachronal waves across all variants."""
    print("\n=== Wave Direction Comparison ===")

    results = {}
    for vkey in ['S', 'F', 'C']:
        vcfg = VARIANTS[vkey]
        model_path = MODELS_DIR / vcfg['model']
        if not model_path.exists():
            continue

        offset = vcfg['phase_offset']
        results[vkey] = {}

        for direction, sign in [('direct', 1.0), ('retrograde', -1.0)]:
            print(f"  {vcfg['name']} {direction} ... ", end='', flush=True)
            t0 = time.time()
            metrics, coherence = run_single(vkey, phase_offset=sign * offset)
            wall = time.time() - t0
            results[vkey][direction] = {
                'speed_mm_s': metrics['mean_speed_mm_s'],
                'step_length_mm': metrics['step_length_mm'],
                'straightness': metrics['straightness'],
                'cot': metrics['cost_of_transport'],
                'coherence': coherence,
            }
            print(f"speed={metrics['mean_speed_mm_s']:.1f} mm/s  "
                  f"straight={metrics['straightness']:.3f}  "
                  f"coherence={coherence:.3f}  ({wall:.1f}s)")

    # save
    with open(Path(outdir) / 'wave_direction.json', 'w') as f:
        json.dump(results, f, indent=2)

    # plot
    fig, axes = plt.subplots(1, 4, figsize=(COL2, 2.4))

    variant_names = []
    metrics_direct = {'speed': [], 'step': [], 'straight': [], 'coherence': []}
    metrics_retro  = {'speed': [], 'step': [], 'straight': [], 'coherence': []}

    for vkey in ['S', 'F', 'C']:
        if vkey not in results:
            continue
        vcfg = VARIANTS[vkey]
        variant_names.append(f"{vcfg['name']}\n({vcfg['n_pairs']}p)")
        d = results[vkey]['direct']
        r = results[vkey]['retrograde']
        metrics_direct['speed'].append(d['speed_mm_s'])
        metrics_direct['step'].append(d['step_length_mm'])
        metrics_direct['straight'].append(d['straightness'])
        metrics_direct['coherence'].append(d['coherence'])
        metrics_retro['speed'].append(r['speed_mm_s'])
        metrics_retro['step'].append(r['step_length_mm'])
        metrics_retro['straight'].append(r['straightness'])
        metrics_retro['coherence'].append(r['coherence'])

    x = np.arange(len(variant_names))
    w = 0.35

    ylabels = ['Speed (mm/s)', 'Step Length (mm)', 'Straightness', 'Wave Coherence']
    keys = ['speed', 'step', 'straight', 'coherence']
    labels = ['a', 'b', 'c', 'd']

    for ax, ylabel, key, lbl in zip(axes, ylabels, keys, labels):
        panel_label(ax, lbl)
        ax.bar(x - w/2, metrics_direct[key], w, color=DIRECT_COLOR,
               label='Direct', edgecolor='black', linewidth=0.4)
        ax.bar(x + w/2, metrics_retro[key], w, color=RETRO_COLOR,
               label='Retrograde', edgecolor='black', linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(variant_names)
        ax.set_ylabel(ylabel)
        if key in ('straight', 'coherence'):
            ax.set_ylim(0, 1.05)
        ax.legend()

    plt.tight_layout()
    save_fig(fig, outdir, 'wave_direction_comparison')

    return results


# 2. Phase noise robustness

def noise_robustness_sweep(outdir):
    """Sweep phase noise amplitude and measure gait degradation.
    Tests the basin of attraction of the Kuramoto-coupled CPG."""
    print("\n=== Phase Noise Robustness Sweep ===")

    noise_levels = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
    n_trials = 3  # average over multiple runs for noisy conditions

    all_results = {}
    for vkey in ['S', 'F', 'C']:
        vcfg = VARIANTS[vkey]
        model_path = MODELS_DIR / vcfg['model']
        if not model_path.exists():
            continue

        print(f"\n  {vcfg['name']}:")
        all_results[vkey] = []

        for noise in noise_levels:
            speeds = []
            straights = []
            coherences = []
            trials = n_trials if noise > 0 else 1

            for trial in range(trials):
                metrics, coherence = run_single(vkey, phase_noise=noise)
                speeds.append(metrics['mean_speed_mm_s'])
                straights.append(metrics['straightness'])
                coherences.append(coherence)

            result = {
                'noise': noise,
                'speed_mean': float(np.mean(speeds)),
                'speed_std': float(np.std(speeds)),
                'straightness_mean': float(np.mean(straights)),
                'straightness_std': float(np.std(straights)),
                'coherence_mean': float(np.mean(coherences)),
                'coherence_std': float(np.std(coherences)),
            }
            all_results[vkey].append(result)
            print(f"    noise={noise:5.1f}  speed={result['speed_mean']:5.1f} +/- {result['speed_std']:4.1f}  "
                  f"straight={result['straightness_mean']:.3f}  "
                  f"coherence={result['coherence_mean']:.3f}")

    # save
    with open(Path(outdir) / 'noise_robustness.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.6))

    for vkey in ['S', 'F', 'C']:
        if vkey not in all_results:
            continue
        pcfg = PVCFG[vkey]
        data = all_results[vkey]
        noise = [d['noise'] for d in data]

        axes[0].errorbar(noise, [d['speed_mean'] for d in data],
                         yerr=[d['speed_std'] for d in data],
                         fmt=f"{pcfg['marker']}-", color=pcfg['color'],
                         label=pcfg['label'])
        axes[1].errorbar(noise, [d['straightness_mean'] for d in data],
                         yerr=[d['straightness_std'] for d in data],
                         fmt=f"{pcfg['marker']}-", color=pcfg['color'],
                         label=pcfg['label'])
        axes[2].errorbar(noise, [d['coherence_mean'] for d in data],
                         yerr=[d['coherence_std'] for d in data],
                         fmt=f"{pcfg['marker']}-", color=pcfg['color'],
                         label=pcfg['label'])

    panel_label(axes[0], 'a')
    axes[0].set_xlabel(r'Phase Noise Amplitude (rad/$\sqrt{s}$)')
    axes[0].set_ylabel('Speed (mm/s)')
    axes[0].legend()

    panel_label(axes[1], 'b')
    axes[1].set_xlabel(r'Phase Noise Amplitude (rad/$\sqrt{s}$)')
    axes[1].set_ylabel('Straightness')
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    panel_label(axes[2], 'c')
    axes[2].set_xlabel(r'Phase Noise Amplitude (rad/$\sqrt{s}$)')
    axes[2].set_ylabel('Wave Coherence')
    axes[2].set_ylim(0, 1.05)
    axes[2].legend()

    plt.tight_layout()
    save_fig(fig, outdir, 'noise_robustness')

    return all_results


# 3. Slope performance

def slope_sweep(outdir):
    """Sweep incline angle and measure locomotion performance.
    Tests morphological adaptation to terrain gradient."""
    print("\n=== Slope Performance Sweep ===")

    angles = [0, 3, 5, 8, 10, 15, 20, 25, 30]

    all_results = {}
    for vkey in ['S', 'F', 'C']:
        vcfg = VARIANTS[vkey]
        model_path = MODELS_DIR / vcfg['model']
        if not model_path.exists():
            continue

        print(f"\n  {vcfg['name']}:")
        all_results[vkey] = []

        for angle in angles:
            print(f"    slope={angle:2d} deg ... ", end='', flush=True)
            t0 = time.time()
            metrics, coherence, raw = run_single(vkey, slope_deg=angle,
                                                  return_results=True)
            wall = time.time() - t0

            # signed forward speed (positive = uphill progress)
            t_raw = raw['t']
            pos_raw = raw['pos']
            mask = t_raw > 2.0
            pos_ss = pos_raw[mask]
            t_ss = t_raw[mask]
            dx = pos_ss[-1, 0] - pos_ss[0, 0]  # signed x displacement
            elapsed = t_ss[-1] - t_ss[0]
            signed_speed = (dx / max(elapsed, 1e-9)) * 1000  # mm/s

            result = {
                'angle_deg': angle,
                'signed_speed_mm_s': float(signed_speed),
                'speed_mm_s': metrics['mean_speed_mm_s'],
                'step_length_mm': metrics['step_length_mm'],
                'straightness': metrics['straightness'],
                'cot': metrics['cost_of_transport'],
                'coherence': coherence,
            }
            all_results[vkey].append(result)
            print(f"fwd_speed={signed_speed:7.1f} mm/s  "
                  f"straight={metrics['straightness']:.3f}  ({wall:.1f}s)")

    # save
    with open(Path(outdir) / 'slope_sweep.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.6))

    for vkey in ['S', 'F', 'C']:
        if vkey not in all_results:
            continue
        pcfg = PVCFG[vkey]
        data = all_results[vkey]
        angs = [d['angle_deg'] for d in data]

        axes[0].plot(angs, [d['signed_speed_mm_s'] for d in data],
                     f"{pcfg['marker']}-", color=pcfg['color'],
                     label=pcfg['label'])
        axes[1].plot(angs, [d['cot'] for d in data],
                     f"{pcfg['marker']}-", color=pcfg['color'],
                     label=pcfg['label'])
        axes[2].plot(angs, [d['straightness'] for d in data],
                     f"{pcfg['marker']}-", color=pcfg['color'],
                     label=pcfg['label'])

    panel_label(axes[0], 'a')
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.6, alpha=0.5)
    axes[0].set_xlabel('Slope Angle (deg)')
    axes[0].set_ylabel('Net Forward Speed (mm/s)')
    axes[0].legend()

    panel_label(axes[1], 'b')
    axes[1].set_xlabel('Slope Angle (deg)')
    axes[1].set_ylabel('Cost of Transport (a.u.)')
    axes[1].legend()

    panel_label(axes[2], 'c')
    axes[2].set_xlabel('Slope Angle (deg)')
    axes[2].set_ylabel('Straightness')
    axes[2].set_ylim(0, 1.05)
    axes[2].legend()

    plt.tight_layout()
    save_fig(fig, outdir, 'slope_performance')

    return all_results


def main():
    parser = argparse.ArgumentParser(description="HALLUCIGENIA Robustness Analysis")
    parser.add_argument('--wave-direction', action='store_true',
                        help='Compare direct vs retrograde metachronal waves')
    parser.add_argument('--noise', action='store_true',
                        help='Phase noise robustness sweep')
    parser.add_argument('--slope', action='store_true',
                        help='Incline angle performance sweep')
    parser.add_argument('--all', action='store_true',
                        help='Run all three analyses')
    parser.add_argument('--outdir', type=str, default=None)
    args = parser.parse_args()

    outdir = args.outdir or str(RESULTS_DIR)
    Path(outdir).mkdir(parents=True, exist_ok=True)

    if args.all or args.wave_direction:
        wave_direction_comparison(outdir)

    if args.all or args.noise:
        noise_robustness_sweep(outdir)

    if args.all or args.slope:
        slope_sweep(outdir)

    if not (args.all or args.wave_direction or args.noise or args.slope):
        print("No analysis selected. Use --all, --wave-direction, --noise, or --slope.")

    print("\nDone.")


if __name__ == "__main__":
    main()
