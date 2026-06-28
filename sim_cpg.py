"""HALLUCIGENIA MuJoCo CPG Locomotion Simulation
   Carmen Mitchell

   Simulates Kuramoto-coupled phase oscillator CPG driving
   pneumatic lobopod legs on a soft-bodied benthic walker.

   Ref: Ijspeert 2008 Neural Networks 21(4):642-653 (CPG review)
        Manton 1950 J Linnean Soc 41:529-570 (Peripatus locomotion)
        Oliveira 2012 PLoS ONE 7(12):e51220 (Onychophora gait)
        Smith & Ortega-Hernandez 2014 Nature 514 (Hallucigenia claws)

   Usage:
       python sim_cpg.py --freq 0.8 --duration 30 --render
       python sim_cpg.py --sweep --duration 20
       python sim_cpg.py --turn-test --duration 15
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import mujoco

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


# CPG defaults (match firmware: locomotion.h)
COUPLING_GAIN = 2.0         # Kuramoto coupling strength
VALVE_DUTY    = 0.4         # fraction of cycle valve is open

# variant configurations
VARIANTS = {
    'S': {  # H. sparsa: 7 leg-pairs, compact worker
        'name': 'HALLU-S', 'species': 'H. sparsa',
        'n_pairs': 7, 'n_valves': 7,
        'phase_offset': np.pi / 7,
        'model': 'hallu_s.xml', 'color': '#1F77B4',
        'body_length_mm': 120,
    },
    'F': {  # H. fortis: 9 leg-pairs, survey variant
        'name': 'HALLU-F', 'species': 'H. fortis',
        'n_pairs': 9, 'n_valves': 9,
        'phase_offset': np.pi / 9,
        'model': 'hallu_f.xml', 'color': '#FF7F0E',
        'body_length_mm': 150,
    },
    'C': {  # Carbotubulus: 25 leg-pairs / 8 valve zones, squad leader
        'name': 'HALLU-C', 'species': 'Carbotubulus',
        'n_pairs': 25, 'n_valves': 8,
        'phase_offset': np.pi / 8,
        'model': 'hallu_c.xml', 'color': '#D62728',
        'body_length_mm': 300,
    },
}


class KuramotoCPG:
    """Kuramoto-coupled phase oscillator CPG for pneumatic valve timing.
    Each oscillator drives one solenoid valve (one leg pair).
    Phase coupling produces a metachronal wave from tail to head."""

    def __init__(self, n_osc, freq_hz, phase_offset, coupling, duty, phase_noise=0.0,
                 init_perturb=0.0):
        self.n = n_osc
        self.freq = freq_hz
        self.offset = phase_offset
        self.K = coupling
        self.duty = duty
        self.phase_noise = phase_noise
        self.init_perturb = init_perturb
        self.threshold = np.cos(np.pi * duty)
        # initialize with staggered phases (tail-to-head wave)
        self.phases = np.array([i * phase_offset for i in range(n_osc)])
        if init_perturb > 0:
            self.phases += init_perturb * np.random.randn(n_osc)

    def step(self, dt, duty_left=None, duty_right=None):
        """Advance oscillators by dt. Returns (left_cmds, right_cmds) in [0,1]."""
        for i in range(self.n):
            coupling = 0.0
            if i > 0:
                coupling += np.sin(self.phases[i-1] - self.phases[i] - self.offset)
            if i < self.n - 1:
                coupling += np.sin(self.phases[i+1] - self.phases[i] + self.offset)
            noise = self.phase_noise * np.sqrt(dt) * np.random.randn() if self.phase_noise > 0 else 0.0
            self.phases[i] += (2 * np.pi * self.freq + self.K * coupling) * dt + noise

        # valve states
        wave = np.sin(self.phases)

        # asymmetric duty for turning
        dl = duty_left if duty_left is not None else self.duty
        dr = duty_right if duty_right is not None else self.duty
        thresh_l = np.cos(np.pi * dl)
        thresh_r = np.cos(np.pi * dr)

        left_cmd  = np.where(wave > thresh_l, 1.0, 0.0)
        right_cmd = np.where(wave > thresh_r, 1.0, 0.0)

        return left_cmd, right_cmd

    def wave_coherence(self):
        """Metachronal wave coherence r in [0,1].
        r=1.0 means perfect wave pattern matching the target offset.
        Adjusts phases by the expected offset before computing Kuramoto r."""
        adjusted = self.phases + np.arange(self.n) * self.offset
        z = np.mean(np.exp(1j * adjusted))
        return float(abs(z))

    def reset(self):
        self.phases = np.array([i * self.offset for i in range(self.n)])


def load_model(model_path):
    """Load MuJoCo model."""
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    return model, data


def get_actuator_mapping(model, variant_key):
    """Map valve indices to actuator IDs.
    For S and F: valve_1/valve_1R, valve_2/valve_2R, ...
    For C: multiple actuators per zone, grouped."""
    vcfg = VARIANTS[variant_key]
    n_valves = vcfg['n_valves']

    if variant_key == 'C':
        # Carbotubulus: 8 valve zones, each controlling 3-4 leg pairs
        # actuators: valve_1a/1aR, valve_1b/1bR, valve_1c/1cR, ... per zone
        zone_legs = [3, 3, 3, 3, 3, 3, 3, 4]  # legs per zone (25 total)
        mapping = []
        for z in range(8):
            zone_left_ids = []
            zone_right_ids = []
            suffixes = ['a', 'b', 'c', 'd']
            for leg_in_zone in range(zone_legs[z]):
                s = suffixes[leg_in_zone]
                l_name = f"valve_{z+1}{s}"
                r_name = f"valve_{z+1}{s}R"
                l_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, l_name)
                r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, r_name)
                if l_id >= 0:
                    zone_left_ids.append(l_id)
                if r_id >= 0:
                    zone_right_ids.append(r_id)
            mapping.append((zone_left_ids, zone_right_ids))
        return mapping
    else:
        mapping = []
        for i in range(1, n_valves + 1):
            left_name = f"valve_{i}"
            right_name = f"valve_{i}R"
            left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, left_name)
            right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, right_name)
            mapping.append((left_id, right_id))
        return mapping


def run_simulation(model, data, cpg, duration, actuator_map, variant_key='S',
                   gait="FORWARD", turn_duty=0.2,
                   record=True):
    """Run CPG-driven locomotion simulation.

    Gaits:
        FORWARD: metachronal wave, symmetric duty
        REVERSE: inverted wave direction
        TURN_LEFT: reduced left duty
        TURN_RIGHT: reduced right duty
        HALT: all valves closed
    """
    dt = model.opt.timestep
    n_steps = int(duration / dt)
    sim_hz = 1.0 / dt

    n_osc = cpg.n

    # recording buffers
    rec_t   = []
    rec_pos = []
    rec_vel = []
    rec_heading = []
    rec_valves = []
    rec_phases = []
    rec_ctrl_effort = []
    rec_coherence = []

    record_interval = int(sim_hz / 50)  # record at 50 Hz

    for step in range(n_steps):
        t = step * dt

        # CPG step (at simulation rate)
        if gait == "HALT":
            left_cmd = np.zeros(n_osc)
            right_cmd = np.zeros(n_osc)
        elif gait == "TURN_LEFT":
            left_cmd, right_cmd = cpg.step(dt, duty_left=turn_duty)
        elif gait == "TURN_RIGHT":
            left_cmd, right_cmd = cpg.step(dt, duty_right=turn_duty)
        else:  # FORWARD
            left_cmd, right_cmd = cpg.step(dt)

        # apply to actuators
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

        # step physics
        mujoco.mj_step(model, data)

        # record
        if record and step % record_interval == 0:
            head_pos = data.sensordata[0:3].copy()
            head_vel = data.sensordata[7:10].copy()
            head_quat = data.sensordata[3:7].copy()

            w, x, y, z = head_quat
            heading = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

            rec_t.append(t)
            rec_pos.append(head_pos)
            rec_vel.append(head_vel)
            rec_heading.append(heading)
            rec_valves.append(np.concatenate([left_cmd, right_cmd]))
            rec_phases.append(cpg.phases.copy() % (2 * np.pi))
            rec_ctrl_effort.append(np.sum(np.abs(data.ctrl)))
            rec_coherence.append(cpg.wave_coherence())

    results = {
        't': np.array(rec_t),
        'pos': np.array(rec_pos),
        'vel': np.array(rec_vel),
        'heading': np.array(rec_heading),
        'valves': np.array(rec_valves),
        'phases': np.array(rec_phases),
        'ctrl_effort': np.array(rec_ctrl_effort),
        'coherence': np.array(rec_coherence),
    }

    return results


def compute_metrics(results):
    """Compute locomotion performance metrics."""
    t = results['t']
    pos = results['pos']
    vel = results['vel']
    heading = results['heading']

    # discard first 2 seconds (transient)
    mask = t > 2.0
    if mask.sum() < 10:
        mask = np.ones(len(t), dtype=bool)

    t_ss = t[mask]
    pos_ss = pos[mask]
    vel_ss = vel[mask]

    # displacement
    dx = pos_ss[-1, 0] - pos_ss[0, 0]
    dy = pos_ss[-1, 1] - pos_ss[0, 1]
    total_disp = np.sqrt(dx**2 + dy**2)
    elapsed = t_ss[-1] - t_ss[0]

    # mean speed
    speed_xy = np.sqrt(vel_ss[:, 0]**2 + vel_ss[:, 1]**2)
    mean_speed = np.mean(speed_xy)

    # mean heading and drift
    mean_heading = np.mean(heading[mask])
    heading_std = np.std(heading[mask])

    # straightness (displacement / path length)
    diffs = np.diff(pos_ss[:, :2], axis=0)
    path_length = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))
    straightness = total_disp / max(path_length, 1e-9)

    # cost of transport (simplified: total actuator effort / distance)
    valves_ss = results['valves'][mask]
    total_effort = np.sum(valves_ss) * (t[1] - t[0]) * (len(t) / len(t_ss))
    cot = total_effort / max(total_disp, 1e-9)

    # count steps (full CPG cycles from valve 1)
    v1 = results['valves'][:, 0]
    edges = np.diff((v1 > 0.5).astype(int))
    n_steps_counted = np.sum(edges > 0)

    return {
        'displacement_m': total_disp,
        'elapsed_s': elapsed,
        'mean_speed_m_s': mean_speed,
        'mean_speed_mm_s': mean_speed * 1000,
        'displacement_speed_m_s': total_disp / max(elapsed, 1e-9),
        'mean_heading_rad': mean_heading,
        'heading_std_rad': heading_std,
        'straightness': straightness,
        'cost_of_transport': cot,
        'step_count': n_steps_counted,
        'step_length_mm': (total_disp / max(n_steps_counted, 1)) * 1000,
    }


def plot_results(results, metrics, freq, outdir):
    """Generate publication-quality plots."""
    if not HAS_PLT:
        print("matplotlib not available, skipping plots")
        return

    from paper_style import apply_style, panel_label, save_fig, COL2, BLUE, GREEN
    apply_style()

    t = results['t']
    pos = results['pos']
    heading = results['heading']
    valves = results['valves']

    fig, axes = plt.subplots(2, 2, figsize=(COL2, COL2 * 0.75))

    # 1. trajectory (top view)
    ax = axes[0, 0]
    panel_label(ax, 'a')
    ax.plot(pos[:, 0] * 1000, pos[:, 1] * 1000, '-', color=BLUE, linewidth=1.0)
    ax.plot(pos[0, 0] * 1000, pos[0, 1] * 1000, 'o', color=GREEN,
            markersize=6, zorder=5, label='Start')
    ax.plot(pos[-1, 0] * 1000, pos[-1, 1] * 1000, 's', color='#D55E00',
            markersize=6, zorder=5, label='End')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_aspect('equal')
    ax.legend()

    # 2. forward displacement vs time
    ax = axes[0, 1]
    panel_label(ax, 'b')
    ax.plot(t, pos[:, 0] * 1000, '-', color=BLUE, linewidth=1.0)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Forward displacement (mm)')

    # 3. valve timing diagram (first 5 seconds)
    ax = axes[1, 0]
    panel_label(ax, 'c')
    mask_5s = t < 5.0
    t_5s = t[mask_5s]
    v_5s = valves[mask_5s, :N_PAIRS]  # left valves only
    cmap = plt.cm.cividis
    for i in range(N_PAIRS):
        ax.fill_between(t_5s, i + v_5s[:, i] * 0.8, i,
                        alpha=0.8, color=cmap(i / max(N_PAIRS - 1, 1)))
        ax.text(-0.3, i + 0.4, f'V{i+1}', fontsize=6, ha='right')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Valve (posterior \u2192 anterior)')
    ax.set_yticks([])
    ax.set_xlim(0, 5)
    ax.grid(False)

    # 4. heading over time
    ax = axes[1, 1]
    panel_label(ax, 'd')
    ax.plot(t, np.degrees(heading), '-', color='#333333', linewidth=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Heading (deg)')
    ax.axhline(np.degrees(metrics['mean_heading_rad']), color='#D55E00',
               linestyle='--', linewidth=0.8, alpha=0.6)

    plt.tight_layout()
    save_fig(fig, outdir, f'hallu_s_cpg_{freq:.1f}Hz')


def plot_frequency_sweep(sweep_results, outdir):
    """Plot locomotion metrics across CPG frequencies."""
    if not HAS_PLT:
        return

    from paper_style import (apply_style, panel_label, save_fig, COL2,
                             BLUE, ORANGE, GREEN)
    apply_style()

    freqs = [r['freq'] for r in sweep_results]
    speeds = [r['metrics']['mean_speed_mm_s'] for r in sweep_results]
    step_lens = [r['metrics']['step_length_mm'] for r in sweep_results]
    straightness = [r['metrics']['straightness'] for r in sweep_results]
    cots = [r['metrics']['cost_of_transport'] for r in sweep_results]

    # biological comparison (Manton 1950)
    bio_freq  = [0.5, 0.8, 1.2, 1.6, 2.0]
    bio_speed = [5.0, 8.0, 13.0, 18.0, 25.0]  # mm/s (Peripatus)

    fig, axes = plt.subplots(2, 2, figsize=(COL2, COL2 * 0.75))

    ax = axes[0, 0]
    panel_label(ax, 'a')
    ax.plot(freqs, speeds, 'o-', color=BLUE, label='HALLU-S (sim)')
    ax.plot(bio_freq, bio_speed, '^--', color=GREEN,
            label='Peripatus (Manton 1950)')
    ax.set_xlabel('CPG Frequency (Hz)')
    ax.set_ylabel('Speed (mm/s)')
    ax.legend()

    ax = axes[0, 1]
    panel_label(ax, 'b')
    ax.plot(freqs, step_lens, 'o-', color=ORANGE)
    ax.axhline(15.0, color='gray', linestyle='--', linewidth=0.6,
               label='Design target')
    ax.set_xlabel('CPG Frequency (Hz)')
    ax.set_ylabel('Step Length (mm)')
    ax.legend()

    ax = axes[1, 0]
    panel_label(ax, 'c')
    ax.plot(freqs, straightness, 'o-', color='#333333')
    ax.set_xlabel('CPG Frequency (Hz)')
    ax.set_ylabel('Straightness (0\u20131)')
    ax.set_ylim(0, 1.05)

    ax = axes[1, 1]
    panel_label(ax, 'd')
    ax.plot(freqs, cots, 's-', color='#D55E00')
    ax.set_xlabel('CPG Frequency (Hz)')
    ax.set_ylabel('Cost of Transport (a.u.)')

    plt.tight_layout()
    save_fig(fig, outdir, 'hallu_s_freq_sweep')


def plot_coupling_sweep(sweep_results, outdir, variant_name):
    """Plot locomotion metrics across coupling gains."""
    if not HAS_PLT:
        return

    from paper_style import (apply_style, panel_label, save_fig, COL2,
                             BLUE, ORANGE)
    apply_style()

    Ks = [r['K'] for r in sweep_results]
    speeds = [r['metrics']['mean_speed_mm_s'] for r in sweep_results]
    step_lens = [r['metrics']['step_length_mm'] for r in sweep_results]
    straightness = [r['metrics']['straightness'] for r in sweep_results]

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.4))

    ax = axes[0]
    panel_label(ax, 'a')
    ax.plot(Ks, speeds, 'o-', color=BLUE)
    ax.axvline(2.0, color='#D55E00', linestyle='--', linewidth=0.8,
               alpha=0.6, label='Design $K$=2.0')
    ax.set_xlabel('Coupling Gain $K$')
    ax.set_ylabel('Speed (mm/s)')
    ax.legend()

    ax = axes[1]
    panel_label(ax, 'b')
    ax.plot(Ks, step_lens, 'o-', color=ORANGE)
    ax.axhline(15.0, color='gray', linestyle='--', linewidth=0.6, alpha=0.5)
    ax.set_xlabel('Coupling Gain $K$')
    ax.set_ylabel('Step Length (mm)')

    ax = axes[2]
    panel_label(ax, 'c')
    ax.plot(Ks, straightness, 'o-', color='#333333')
    ax.axvline(2.0, color='#D55E00', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xlabel('Coupling Gain $K$')
    ax.set_ylabel('Straightness (0\u20131)')
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    save_fig(fig, outdir, f'{variant_name.lower()}_coupling_sweep')


def plot_variant_comparison(all_variant_results, outdir):
    """Plot speed comparison across all variants."""
    if not HAS_PLT:
        return

    from paper_style import (apply_style, panel_label, save_fig, COL2,
                             VCFG as PVCFG, GREEN)
    apply_style()

    bio_freq  = [0.5, 0.8, 1.2, 1.6, 2.0]
    bio_speed = [5.0, 8.0, 13.0, 18.0, 25.0]

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.6))

    for vkey, vdata in all_variant_results.items():
        pcfg = PVCFG[vkey]
        freqs = [r['freq'] for r in vdata]
        speeds = [r['metrics']['mean_speed_mm_s'] for r in vdata]
        step_lens = [r['metrics']['step_length_mm'] for r in vdata]
        straightness = [r['metrics']['straightness'] for r in vdata]

        axes[0].plot(freqs, speeds, f"{pcfg['marker']}-", color=pcfg['color'],
                     label=pcfg['label'])
        axes[1].plot(freqs, step_lens, f"{pcfg['marker']}-", color=pcfg['color'],
                     label=pcfg['label'])
        axes[2].plot(freqs, straightness, f"{pcfg['marker']}-", color=pcfg['color'],
                     label=pcfg['label'])

    axes[0].plot(bio_freq, bio_speed, '^--', color=GREEN,
                 label='Peripatus (Manton 1950)')

    panel_label(axes[0], 'a')
    axes[0].set_xlabel('CPG Frequency (Hz)')
    axes[0].set_ylabel('Speed (mm/s)')
    axes[0].legend()

    panel_label(axes[1], 'b')
    axes[1].axhline(15.0, color='gray', linestyle='--', linewidth=0.6)
    axes[1].set_xlabel('CPG Frequency (Hz)')
    axes[1].set_ylabel('Step Length (mm)')
    axes[1].legend()

    panel_label(axes[2], 'c')
    axes[2].set_xlabel('CPG Frequency (Hz)')
    axes[2].set_ylabel('Straightness (0\u20131)')
    axes[2].set_ylim(0, 1.05)
    axes[2].legend()

    plt.tight_layout()
    save_fig(fig, outdir, 'variant_comparison')


def main():
    parser = argparse.ArgumentParser(description="HALLUCIGENIA MuJoCo CPG Simulation")
    parser.add_argument('--freq', type=float, default=0.8,
                        help='CPG frequency in Hz (default: 0.8)')
    parser.add_argument('--duration', type=float, default=20.0,
                        help='Simulation duration in seconds (default: 20)')
    parser.add_argument('--sweep', action='store_true',
                        help='Run frequency sweep (0.3-2.5 Hz)')
    parser.add_argument('--turn-test', action='store_true',
                        help='Run turning characterization')
    parser.add_argument('--coupling-sweep', action='store_true',
                        help='Run coupling gain sweep (K=0.5 to 6.0)')
    parser.add_argument('--friction-sweep', action='store_true',
                        help='Run substrate friction sweep')
    parser.add_argument('--variant', type=str, default='S',
                        choices=['S', 'F', 'C'],
                        help='Robot variant (S=sparsa 7-pair, F=fortis 9-pair, C=Carbotubulus 8-zone)')
    parser.add_argument('--coupling', type=float, default=COUPLING_GAIN,
                        help='Kuramoto coupling gain (default: 2.0)')
    parser.add_argument('--duty', type=float, default=VALVE_DUTY,
                        help='Valve duty cycle (default: 0.4)')
    parser.add_argument('--render', action='store_true',
                        help='Open viewer (requires display)')
    parser.add_argument('--outdir', type=str, default=None)
    args = parser.parse_args()

    vcfg = VARIANTS[args.variant]
    model_path = Path(__file__).parent / 'models' / vcfg['model']
    outdir = args.outdir or str(Path(__file__).parent / 'results')
    Path(outdir).mkdir(parents=True, exist_ok=True)

    print(f"HALLUCIGENIA {vcfg['name']} ({vcfg['species']}) MuJoCo CPG Simulation")
    print(f"  Model: {model_path}")
    print(f"  Leg pairs: {vcfg['n_pairs']}, Valves: {vcfg['n_valves']}")
    print(f"  MuJoCo: {mujoco.__version__}")
    print()

    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        return

    model, data = load_model(str(model_path))
    actuator_map = get_actuator_mapping(model, args.variant)
    print(f"  Actuators mapped: {len(actuator_map)} valve groups")
    print(f"  Timestep: {model.opt.timestep} s")
    print()

    n_valves = vcfg['n_valves']
    phase_offset = vcfg['phase_offset']
    vname = vcfg['name']

    def _jsonable(d):
        return {k: float(v) if hasattr(v, 'item') else v for k, v in d.items()}

    if args.sweep:
        print("=== Frequency Sweep ===")
        freqs = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
        sweep_results = []

        for f in freqs:
            mujoco.mj_resetData(model, data)
            cpg = KuramotoCPG(n_valves, f, phase_offset, args.coupling, args.duty)
            print(f"  freq={f:.1f} Hz ... ", end='', flush=True)

            t0 = time.time()
            results = run_simulation(model, data, cpg, args.duration, actuator_map,
                                     variant_key=args.variant)
            wall = time.time() - t0
            metrics = compute_metrics(results)
            sweep_results.append({'freq': f, 'metrics': metrics, 'results': results})
            print(f"speed={metrics['mean_speed_mm_s']:.1f} mm/s  "
                  f"step={metrics['step_length_mm']:.1f} mm  "
                  f"straight={metrics['straightness']:.3f}  "
                  f"({wall:.1f}s wall)")

        plot_frequency_sweep(sweep_results, outdir)
        summary = [dict(_jsonable(sr['metrics']), freq_hz=sr['freq']) for sr in sweep_results]
        fname = f'{vname.lower()}_freq_sweep.json'
        with open(Path(outdir) / fname, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nMetrics saved: {Path(outdir) / fname}")

    elif args.coupling_sweep:
        print("=== Coupling Gain Sweep ===")
        Ks = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
        sweep_results = []

        for K in Ks:
            mujoco.mj_resetData(model, data)
            cpg = KuramotoCPG(n_valves, args.freq, phase_offset, K, args.duty)
            print(f"  K={K:.1f} ... ", end='', flush=True)

            t0 = time.time()
            results = run_simulation(model, data, cpg, args.duration, actuator_map,
                                     variant_key=args.variant)
            wall = time.time() - t0
            metrics = compute_metrics(results)
            sweep_results.append({'K': K, 'metrics': metrics})
            print(f"speed={metrics['mean_speed_mm_s']:.1f} mm/s  "
                  f"step={metrics['step_length_mm']:.1f} mm  "
                  f"straight={metrics['straightness']:.3f}  "
                  f"({wall:.1f}s wall)")

        plot_coupling_sweep(sweep_results, outdir, vname)
        summary = [dict(_jsonable(sr['metrics']), K=sr['K']) for sr in sweep_results]
        with open(Path(outdir) / f'{vname.lower()}_coupling_sweep.json', 'w') as f:
            json.dump(summary, f, indent=2)

    elif args.friction_sweep:
        print("=== Substrate Friction Sweep ===")
        # hard rock, gravel, sand, soft mud
        substrates = [
            ('hard_rock', 1.5), ('gravel', 1.0), ('sand', 0.6),
            ('soft_mud', 0.3), ('silt', 0.15),
        ]
        sweep_results = []

        for sub_name, mu in substrates:
            mujoco.mj_resetData(model, data)
            # modify floor friction
            floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
            model.geom_friction[floor_id, 0] = mu

            cpg = KuramotoCPG(n_valves, args.freq, phase_offset, args.coupling, args.duty)
            print(f"  {sub_name} (mu={mu:.2f}) ... ", end='', flush=True)

            results = run_simulation(model, data, cpg, args.duration, actuator_map,
                                     variant_key=args.variant)
            metrics = compute_metrics(results)
            sweep_results.append({
                'substrate': sub_name, 'friction': mu, 'metrics': metrics
            })
            print(f"speed={metrics['mean_speed_mm_s']:.1f} mm/s  "
                  f"straight={metrics['straightness']:.3f}")

        # restore floor friction
        model.geom_friction[floor_id, 0] = 1.0

        with open(Path(outdir) / f'{vname.lower()}_friction_sweep.json', 'w') as f:
            json.dump([dict(_jsonable(sr['metrics']),
                           substrate=sr['substrate'], friction=sr['friction'])
                       for sr in sweep_results], f, indent=2)
        print(f"\nFriction data saved")

    elif args.turn_test:
        print("=== Turning Characterization ===")
        turn_duties = [0.1, 0.15, 0.2, 0.25, 0.3]
        turn_results = []

        for td in turn_duties:
            mujoco.mj_resetData(model, data)
            cpg = KuramotoCPG(n_valves, args.freq, phase_offset, args.coupling, args.duty)
            print(f"  turn_duty={td:.2f} ... ", end='', flush=True)

            results = run_simulation(model, data, cpg, args.duration, actuator_map,
                                     variant_key=args.variant,
                                     gait="TURN_LEFT", turn_duty=td)
            metrics = compute_metrics(results)
            heading_change = results['heading'][-1] - results['heading'][0]
            turn_rate = heading_change / max(results['t'][-1], 1e-9)

            turn_results.append({
                'turn_duty': td,
                'heading_change_deg': float(np.degrees(heading_change)),
                'turn_rate_deg_s': float(np.degrees(turn_rate)),
                'speed_mm_s': float(metrics['mean_speed_mm_s']),
            })
            print(f"turn_rate={np.degrees(turn_rate):.1f} deg/s  "
                  f"heading_change={np.degrees(heading_change):.1f} deg  "
                  f"speed={metrics['mean_speed_mm_s']:.1f} mm/s")

        with open(Path(outdir) / f'{vname.lower()}_turn_test.json', 'w') as f:
            json.dump(turn_results, f, indent=2)
        print(f"\nTurn data saved")

    else:
        print(f"=== Single Run: {args.freq:.1f} Hz, {args.duration:.0f} s ===")
        cpg = KuramotoCPG(n_valves, args.freq, phase_offset, args.coupling, args.duty)

        t0 = time.time()
        results = run_simulation(model, data, cpg, args.duration, actuator_map,
                                 variant_key=args.variant)
        wall = time.time() - t0
        metrics = compute_metrics(results)

        print(f"\nResults:")
        print(f"  Displacement:     {metrics['displacement_m']*1000:.1f} mm")
        print(f"  Mean speed:       {metrics['mean_speed_mm_s']:.1f} mm/s")
        print(f"  Step count:       {metrics['step_count']}")
        print(f"  Step length:      {metrics['step_length_mm']:.1f} mm")
        print(f"  Straightness:     {metrics['straightness']:.3f}")
        print(f"  Heading std:      {np.degrees(metrics['heading_std_rad']):.1f} deg")
        print(f"  Cost of transport:{metrics['cost_of_transport']:.1f}")
        print(f"  Wall time:        {wall:.1f} s ({args.duration/wall:.0f}x realtime)")

        plot_results(results, metrics, args.freq, outdir)

        with open(Path(outdir) / f'{vname.lower()}_run_{args.freq:.1f}Hz.json', 'w') as f:
            json.dump(_jsonable(metrics), f, indent=2)


if __name__ == "__main__":
    main()
