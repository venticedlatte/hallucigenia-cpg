"""HALLUCIGENIA Gait Analysis
   Carmen Mitchell

   Phase-lag analysis, normalized cost of transport, and gait cycle
   visualization for the CPG locomotion simulation.

   Ref: Ijspeert 2008 Neural Networks 21(4):642-653 (CPG review)
        Oliveira 2012 PLoS ONE 7(12):e51220 (Onychophora phase lag)
        Manton 1950 J Linnean Soc 41:529-570 (Peripatus locomotion)

   Usage:
       python analyze_gait.py
       python analyze_gait.py --variant F
       python analyze_gait.py --all-variants
"""

import argparse
import json
from pathlib import Path

import numpy as np
import mujoco

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import sim_cpg components
from sim_cpg import (VARIANTS, COUPLING_GAIN, VALVE_DUTY,
                     KuramotoCPG, load_model, get_actuator_mapping,
                     run_simulation, compute_metrics)

from paper_style import (apply_style, panel_label, save_fig, COL2,
                         VCFG as PVCFG, BLUE, ORANGE, PINK, GREEN, RED,
                         GAIT_CMAP)
apply_style()

RESULTS_DIR = Path(__file__).parent / 'results'
MODELS_DIR  = Path(__file__).parent / 'models'

def variant_key_from_name(name):
    """Map variant name back to key."""
    for k, v in VARIANTS.items():
        if v['name'] == name:
            return k
    return 'S'


def phase_lag_analysis(results, vcfg, outdir):
    """Compute actual inter-oscillator phase lag and compare to target.
    Measures phase differences between adjacent valve oscillators
    during steady-state walking."""
    t = results['t']
    phases = results['phases']
    n_valves = vcfg['n_valves']
    target_offset = vcfg['phase_offset']

    # steady-state: discard first 3 seconds
    mask = t > 3.0
    t_ss = t[mask]
    ph_ss = phases[mask]

    # compute inter-oscillator phase differences
    # delta_phi[i] = phase[i+1] - phase[i] (mod 2pi, centered on [-pi, pi])
    n_pairs = n_valves - 1
    deltas = np.zeros((len(t_ss), n_pairs))
    for i in range(n_pairs):
        raw = ph_ss[:, i+1] - ph_ss[:, i]
        deltas[:, i] = (raw + np.pi) % (2 * np.pi) - np.pi

    # statistics
    mean_deltas = np.mean(deltas, axis=0)
    std_deltas = np.std(deltas, axis=0)
    global_mean = np.mean(mean_deltas)
    global_std = np.std(mean_deltas)

    # target comparison
    # Kuramoto coupling drives toward phases[i+1] - phases[i] = -offset
    # (anterior-to-posterior wave: V1 fires first, wave sweeps to tail)
    coupling_target = -target_offset
    target_centered = ((coupling_target + np.pi) % (2 * np.pi)) - np.pi
    errors = mean_deltas - target_centered
    rmse = np.sqrt(np.mean(errors**2))

    print(f"\n  Phase-Lag Analysis ({vcfg['name']}):")
    print(f"    Target phase lag: {np.degrees(coupling_target):.1f} deg "
          f"(anterior-to-posterior metachronal wave)")
    print(f"    Mean measured lag: {np.degrees(global_mean):.1f} deg")
    print(f"    Lag std across pairs: {np.degrees(global_std):.1f} deg")
    print(f"    RMSE vs target: {np.degrees(rmse):.1f} deg")
    print(f"    Per-pair lags (deg): {', '.join(f'{np.degrees(d):.1f}' for d in mean_deltas)}")

    # plot
    pcfg = PVCFG.get(variant_key_from_name(vcfg['name']), {})
    vcolor = pcfg.get('color', BLUE)

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.4))

    # (a) phase lag per pair
    ax = axes[0]
    panel_label(ax, 'a')
    pairs = np.arange(1, n_pairs + 1)
    ax.bar(pairs, np.degrees(mean_deltas), yerr=np.degrees(std_deltas),
           color=vcolor, alpha=0.85, capsize=2, edgecolor='black', linewidth=0.4)
    ax.axhline(np.degrees(target_centered), color=RED, linestyle='--',
               linewidth=1.0, label=f'Target ({np.degrees(coupling_target):.1f}\u00b0)')
    ax.set_xlabel('Adjacent Pair Index')
    ax.set_ylabel('Phase Lag (deg)')
    ax.legend()

    # (b) phase trajectory over time (first 5 oscillators)
    ax = axes[1]
    panel_label(ax, 'b')
    n_show = min(5, n_valves)
    colors = plt.cm.cividis(np.linspace(0.1, 0.9, n_show))
    t_plot = t_ss[t_ss < 8.0]
    ph_plot = ph_ss[:len(t_plot)]
    for i in range(n_show):
        ax.plot(t_plot, np.degrees(ph_plot[:, i] % (2*np.pi)),
                '.', color=colors[i], markersize=1, alpha=0.5, label=f'V{i+1}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Phase (deg)')
    ax.legend(markerscale=5)
    ax.set_ylim(0, 360)

    # (c) phase portrait: V1 vs V2
    ax = axes[2]
    panel_label(ax, 'c')
    if n_valves >= 3:
        ax.scatter(np.degrees(ph_ss[:, 0] % (2*np.pi)),
                   np.degrees(ph_ss[:, 1] % (2*np.pi)),
                   c=t_ss, cmap='cividis', s=1.5, alpha=0.5)
        ax.set_xlabel('V1 Phase (deg)')
        ax.set_ylabel('V2 Phase (deg)')
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 360)
        ax.set_aspect('equal')
        ideal_x = np.linspace(0, 360, 100)
        ideal_y = (ideal_x + np.degrees(coupling_target)) % 360
        ax.plot(ideal_x, ideal_y, '--', color=RED, linewidth=0.8, alpha=0.6,
                label='Ideal')
        ax.legend()
        ax.grid(False)

    plt.tight_layout()
    save_fig(fig, outdir, f'{vcfg["name"].lower()}_phase_analysis')

    return {
        'target_lag_deg': float(np.degrees(coupling_target)),
        'mean_lag_deg': float(np.degrees(global_mean)),
        'std_lag_deg': float(np.degrees(global_std)),
        'rmse_deg': float(np.degrees(rmse)),
        'per_pair_deg': [float(np.degrees(d)) for d in mean_deltas],
        'wave_direction': 'anterior-to-posterior',
    }


def energy_analysis(results, model, vcfg, outdir):
    """Compute normalized cost of transport.
    CoT = P / (m * g * v), dimensionless.
    P = mean control effort (proxy for pneumatic power).
    """
    t = results['t']
    pos = results['pos']
    ctrl = results['ctrl_effort']

    mask = t > 2.0
    t_ss = t[mask]
    pos_ss = pos[mask]
    ctrl_ss = ctrl[mask]

    # body mass from model
    total_mass = sum(model.body_mass)
    g = 9.81

    # displacement speed
    dx = pos_ss[-1, 0] - pos_ss[0, 0]
    dy = pos_ss[-1, 1] - pos_ss[0, 1]
    disp = np.sqrt(dx**2 + dy**2)
    elapsed = t_ss[-1] - t_ss[0]
    v_disp = disp / max(elapsed, 1e-9)

    # mean instantaneous speed
    speed_xy = np.sqrt(np.diff(pos_ss[:, 0])**2 + np.diff(pos_ss[:, 1])**2)
    dt_rec = np.diff(t_ss)
    v_inst = speed_xy / dt_rec
    v_mean = np.mean(v_inst)

    # power proxy: mean total control magnitude per timestep
    mean_ctrl = np.mean(ctrl_ss)

    # normalized CoT (dimensionless)
    # using displacement speed for consistency
    cot_normalized = mean_ctrl / (total_mass * g * max(v_disp, 1e-9))

    # Froude number (dimensionless speed): Fr = v^2 / (g * L)
    body_len = vcfg['body_length_mm'] / 1000.0
    froude = v_disp**2 / (g * body_len)

    # specific resistance (another CoT normalization)
    specific_resistance = mean_ctrl / (total_mass * max(v_disp, 1e-9))

    print(f"\n  Energy Analysis ({vcfg['name']}):")
    print(f"    Body mass: {total_mass*1000:.1f} g")
    print(f"    Body length: {vcfg['body_length_mm']} mm")
    print(f"    Displacement speed: {v_disp*1000:.1f} mm/s")
    print(f"    Mean speed: {v_mean*1000:.1f} mm/s")
    print(f"    Mean control effort: {mean_ctrl:.3f}")
    print(f"    Normalized CoT: {cot_normalized:.4f}")
    print(f"    Specific resistance: {specific_resistance:.4f}")
    print(f"    Froude number: {froude:.6f}")

    return {
        'mass_g': float(total_mass * 1000),
        'body_length_mm': vcfg['body_length_mm'],
        'displacement_speed_mm_s': float(v_disp * 1000),
        'mean_speed_mm_s': float(v_mean * 1000),
        'mean_ctrl_effort': float(mean_ctrl),
        'cot_normalized': float(cot_normalized),
        'specific_resistance': float(specific_resistance),
        'froude_number': float(froude),
    }


def render_gait_frames(model, data, cpg, actuator_map, variant_key, vcfg, outdir):
    """Render side-view snapshots at evenly spaced phases of one walking cycle.
    Captures 8 frames spanning one full CPG cycle."""
    n_frames = 8
    freq = 0.8
    cycle_time = 1.0 / freq
    dt = model.opt.timestep
    n_valves = vcfg['n_valves']

    # settle for 3 seconds first
    settle_steps = int(3.0 / dt)
    mujoco.mj_resetData(model, data)
    for _ in range(settle_steps):
        left_cmd, right_cmd = cpg.step(dt)
        if variant_key == 'C':
            for i, (l_ids, r_ids) in enumerate(actuator_map):
                for l_id in l_ids:
                    data.ctrl[l_id] = left_cmd[i]
                for r_id in r_ids:
                    data.ctrl[r_id] = right_cmd[i]
        else:
            for i, (l_id, r_id) in enumerate(actuator_map):
                data.ctrl[l_id] = left_cmd[i]
                data.ctrl[r_id] = right_cmd[i]
        mujoco.mj_step(model, data)

    # set up offscreen renderer
    width, height = 640, 240
    renderer = mujoco.Renderer(model, height, width)

    # camera: side view, tracking the robot
    head_pos = data.sensordata[0:3].copy()

    frames = []
    frame_valves = []
    frame_phases = []
    steps_per_frame = int(cycle_time / n_frames / dt)

    for f_idx in range(n_frames):
        for _ in range(steps_per_frame):
            left_cmd, right_cmd = cpg.step(dt)
            if variant_key == 'C':
                for i, (l_ids, r_ids) in enumerate(actuator_map):
                    for l_id in l_ids:
                        data.ctrl[l_id] = left_cmd[i]
                    for r_id in r_ids:
                        data.ctrl[r_id] = right_cmd[i]
            else:
                for i, (l_id, r_id) in enumerate(actuator_map):
                    data.ctrl[l_id] = left_cmd[i]
                    data.ctrl[r_id] = right_cmd[i]
            mujoco.mj_step(model, data)

        # render
        cur_pos = data.sensordata[0:3].copy()
        renderer.update_scene(data)
        # adjust camera to track robot from the side
        renderer.scene.camera[0].lookat[:] = cur_pos
        renderer.scene.camera[0].distance = 0.15
        renderer.scene.camera[0].azimuth = 90  # side view
        renderer.scene.camera[0].elevation = -15

        img = renderer.render()
        frames.append(img.copy())
        frame_valves.append(left_cmd.copy())
        frame_phases.append(cpg.phases.copy() % (2 * np.pi))

    renderer.close()

    # compose into a single figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    fig.suptitle(f'{vcfg["name"]} Walking Cycle | One CPG Period at 0.8 Hz',
                 fontsize=11)

    for i in range(n_frames):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        ax.imshow(frames[i])
        ax.set_title(f't = {i}/{n_frames}T', fontsize=9)
        ax.axis('off')

        # add valve state bar below each frame
        v = frame_valves[i]
        bar_height = frames[i].shape[0] // 15
        bar = np.zeros((bar_height, frames[i].shape[1], 3), dtype=np.uint8)
        valve_width = frames[i].shape[1] // len(v)
        for vi in range(len(v)):
            color = [50, 200, 50] if v[vi] > 0.5 else [80, 80, 80]
            bar[:, vi*valve_width:(vi+1)*valve_width] = color
        ax.imshow(bar, extent=[0, frames[i].shape[1],
                               frames[i].shape[0] + bar_height, frames[i].shape[0]],
                  aspect='auto')

    plt.tight_layout()
    outpath = Path(outdir) / f'{vcfg["name"].lower()}_gait_cycle.png'
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Gait cycle frames saved: {outpath}")

    return frames


def render_gait_diagram(results, vcfg, outdir):
    """Generate a gait diagram: horizontal bars showing valve ON periods
    over several walking cycles. Standard locomotion analysis figure."""
    t = results['t']
    valves = results['valves']
    n_valves = vcfg['n_valves']

    # show 4 complete cycles at 0.8 Hz = 5 seconds
    mask = (t > 3.0) & (t < 8.0)
    t_diag = t[mask] - t[mask][0]
    v_diag = valves[mask, :n_valves]  # left valves only

    fig_h = 1.8 + n_valves * 0.28
    fig, ax = plt.subplots(figsize=(COL2, fig_h))

    colors = plt.cm.cividis(np.linspace(0.1, 0.9, n_valves))

    for i in range(n_valves):
        y_base = n_valves - 1 - i
        # find ON periods
        on = v_diag[:, i] > 0.5
        starts = []
        ends = []
        in_on = False
        for k in range(len(on)):
            if on[k] and not in_on:
                starts.append(t_diag[k])
                in_on = True
            elif not on[k] and in_on:
                ends.append(t_diag[k])
                in_on = False
        if in_on:
            ends.append(t_diag[-1])

        for s, e in zip(starts, ends):
            ax.barh(y_base, e - s, left=s, height=0.65,
                    color=colors[i], edgecolor='black', linewidth=0.3, alpha=0.9)

        ax.text(-0.12, y_base, f'V{i+1}', ha='right', va='center', fontsize=7)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Valve (posterior \u2192 anterior)')
    ax.set_yticks([])
    ax.set_xlim(0, t_diag[-1])
    ax.set_ylim(-0.5, n_valves - 0.5)
    ax.grid(False)

    # mark expected wave period
    period = 1.0 / 0.8
    for p in range(1, 5):
        ax.axvline(p * period, color='gray', linestyle=':', linewidth=0.4, alpha=0.4)

    plt.tight_layout()
    save_fig(fig, outdir, f'{vcfg["name"].lower()}_gait_diagram')


def run_variant_analysis(variant_key, outdir):
    """Run full gait analysis for one variant."""
    vcfg = VARIANTS[variant_key]
    model_path = MODELS_DIR / vcfg['model']

    print(f"\n{'='*60}")
    print(f"  {vcfg['name']} ({vcfg['species']}) Gait Analysis")
    print(f"{'='*60}")

    if not model_path.exists():
        print(f"  Model not found: {model_path}")
        return None

    model, data = load_model(str(model_path))
    actuator_map = get_actuator_mapping(model, variant_key)
    n_valves = vcfg['n_valves']

    # run simulation with phase recording
    cpg = KuramotoCPG(n_valves, 0.8, vcfg['phase_offset'], COUPLING_GAIN, VALVE_DUTY)
    results = run_simulation(model, data, cpg, 20.0, actuator_map,
                             variant_key=variant_key)
    metrics = compute_metrics(results)

    # 1. phase-lag analysis
    phase_data = phase_lag_analysis(results, vcfg, outdir)

    # 2. energy analysis
    energy_data = energy_analysis(results, model, vcfg, outdir)

    # 3. gait diagram
    render_gait_diagram(results, vcfg, outdir)

    # 4. gait cycle frames (rendered snapshots)
    print(f"\n  Rendering gait cycle frames...")
    mujoco.mj_resetData(model, data)
    cpg2 = KuramotoCPG(n_valves, 0.8, vcfg['phase_offset'], COUPLING_GAIN, VALVE_DUTY)
    try:
        render_gait_frames(model, data, cpg2, actuator_map, variant_key, vcfg, outdir)
    except Exception as e:
        print(f"    Rendering failed (no display?): {e}")
        print(f"    Gait diagram and phase analysis still saved.")

    # save combined analysis
    analysis = {
        'variant': variant_key,
        'name': vcfg['name'],
        'species': vcfg['species'],
        'n_pairs': vcfg['n_pairs'],
        'n_valves': n_valves,
        'phase_lag': phase_data,
        'energy': energy_data,
        'locomotion': {
            'displacement_mm': float(metrics['displacement_m'] * 1000),
            'mean_speed_mm_s': float(metrics['mean_speed_mm_s']),
            'step_count': int(metrics['step_count']),
            'step_length_mm': float(metrics['step_length_mm']),
            'straightness': float(metrics['straightness']),
        },
    }

    with open(Path(outdir) / f'{vcfg["name"].lower()}_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)

    return analysis


def plot_energy_comparison(analyses, outdir):
    """Compare energy metrics across variants."""
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.4))

    names = []
    cots = []
    frs = []
    masses = []
    colors_list = []

    for a in analyses:
        vkey = a['variant']
        pcfg = PVCFG.get(vkey, {})
        names.append(f"{a['name']}\n({a['n_pairs']}p)")
        cots.append(a['energy']['cot_normalized'])
        frs.append(a['energy']['froude_number'])
        masses.append(a['energy']['mass_g'])
        colors_list.append(pcfg.get('color', BLUE))

    x = np.arange(len(names))

    axes[0].bar(x, cots, color=colors_list, edgecolor='black', linewidth=0.4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel('Normalized CoT')
    panel_label(axes[0], 'a')

    axes[1].bar(x, frs, color=colors_list, edgecolor='black', linewidth=0.4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel('Froude Number')
    panel_label(axes[1], 'b')

    axes[2].bar(x, masses, color=colors_list, edgecolor='black', linewidth=0.4)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names)
    axes[2].set_ylabel('Mass (g)')
    panel_label(axes[2], 'c')

    plt.tight_layout()
    save_fig(fig, outdir, 'energy_comparison')


def main():
    parser = argparse.ArgumentParser(description="HALLUCIGENIA Gait Analysis")
    parser.add_argument('--variant', type=str, default='S',
                        choices=['S', 'F', 'C'])
    parser.add_argument('--all-variants', action='store_true',
                        help='Run analysis on all available variants')
    parser.add_argument('--outdir', type=str, default=None)
    args = parser.parse_args()

    outdir = args.outdir or str(RESULTS_DIR)
    Path(outdir).mkdir(parents=True, exist_ok=True)

    if args.all_variants:
        analyses = []
        for vkey in ['S', 'F', 'C']:
            model_path = MODELS_DIR / VARIANTS[vkey]['model']
            if model_path.exists():
                a = run_variant_analysis(vkey, outdir)
                if a:
                    analyses.append(a)
        if len(analyses) > 1:
            plot_energy_comparison(analyses, outdir)
    else:
        run_variant_analysis(args.variant, outdir)


if __name__ == "__main__":
    main()
