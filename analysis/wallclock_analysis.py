"""
Wall-clock time based epiplexity analysis for Track 1 (modded-nanogpt speedrun).

Track 1's goal is to reduce wall clock time to reach 3.28 val loss.
The original epiplexity used step number as the integration variable:
    S_step = ∫[L(step) - L_final] d(step)

This script computes time-based epiplexity:
    S_time = ∫[L(t) - L_final] dt,  where t is wall clock time in seconds

and compares the two to see which better captures innovation.
"""
import matplotlib
matplotlib.use('Agg')

import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from scipy import stats

TRACK1_DIR = Path(os.environ.get(
    "TRACK1_DIR",
    os.path.expanduser("~/modded-nanogpt-ref/records/track_1_short")
))
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ─── Category classification (same as parse_track1.py) ──────────────────────

ARCHITECTURE_KEYWORDS = {
    'ModernArch', 'ValueEmbed', 'SkipMLPBlocks', 'SparseAttnGate', 'Smear',
    'WindowWarmup', 'MultiTokenPrediction', 'PairedHeadAttention',
    'BigramHashEmbedding', 'UntieEmbed', 'SoftCap', 'ParallelResiduals',
    'PartialKeyOffset', 'VeSkipGates', 'LogitRescale', 'Backout',
    'UNetDoubleLr', 'ScaleShortcuts', 'UNetValueEmbedsTweaks',
    'SparsifyEmbeds', 'DropAttn', 'ScaleUp1B', 'RefineSkip',
    'RetieLMHead', 'UntieValueEmbeddings', 'ImprovedLMHead',
    'MimeticValueOutput', 'BosAlign', 'SALambdaOnWeights',
    'ShortcutsTweaks', 'QuantizedFP4', 'Fp8LmHead',
}

OPTIMIZATION_KEYWORDS = {
    'Muon', 'NorMuon', 'BatchSize', 'BatchSizeSchedule', 'CautiousWD',
    'AdamSyncGradientHook', 'MixedPrecisionInterweavedOptimizer',
    'UnifiedOptimizers', 'FixMuonLR', 'MuonCustomSizing', 'PolarExpress',
    'SmoothedScalars', 'GatesToCompiledAdam', 'SOAP', 'Optimizers',
    'AdamW', 'llmc', 'MuonWithAuxAdamExample', 'Sub3Min', 'RuleTweak',
    'CautiousWDAdam', 'NorMuonOptimsAndFixes', 'Replicateleloykun',
    '50Bruns',
}

ENGINEERING_KEYWORDS = {
    'DistributedMuon', 'PyTorch25', 'CastBf16', 'FlexAttention', 'FA3',
    'FasterReduce', 'StableTorch', 'TritonMuon', 'FusedLinearReLUSquare',
    'FusedSoftcappedEntropy', 'CrossEntropyKernel', 'FlattenForward',
    'TransposeCopyBackward', 'KernelTuning', 'VeFused',
    'SparseBigramGradient', 'SimplifyHC', 'VarlenMaxDocs',
    'FuseCEFwdAndBwd', 'PairedHeadMuon', 'MFUTweaks', 'EvenFasterReduce',
    'noallreduce', 'UpgradeTorch190', 'Yarn', 'VectSigmoidBFloat16',
    'AsyncDataLoadAttnFinalWindow', 'BF16CE', 'CustomBatching',
    'BigramHashH2D', 'VeTuned', 'ShortWindow',
}


def classify_submission(name: str) -> str:
    parts = re.split(r'^\d{4}-\d{2}-\d{2}[-_]', name)
    label = parts[-1] if len(parts) > 1 else name
    if label in ARCHITECTURE_KEYWORDS:
        return 'Architecture'
    if label in OPTIMIZATION_KEYWORDS:
        return 'Optimization'
    if label in ENGINEERING_KEYWORDS:
        return 'Engineering'
    for kw in ARCHITECTURE_KEYWORDS:
        if kw.lower() in label.lower():
            return 'Architecture'
    for kw in OPTIMIZATION_KEYWORDS:
        if kw.lower() in label.lower():
            return 'Optimization'
    for kw in ENGINEERING_KEYWORDS:
        if kw.lower() in label.lower():
            return 'Engineering'
    return 'Optimization'


def parse_date(dirname: str) -> str:
    m = re.match(r'(\d{4}-\d{2}-\d{2})', dirname)
    return m.group(1) if m else '1970-01-01'


def extract_label(dirname: str) -> str:
    parts = re.split(r'^\d{4}-\d{2}-\d{2}[-_]', dirname)
    return parts[-1] if len(parts) > 1 else dirname


# ─── Log parsing with train_time ────────────────────────────────────────────

def parse_modern_log_with_time(filepath: Path):
    """
    Parse modern format lines with val_loss AND train_time:
      step:N/TOTAL val_loss:X.XXXX train_time:NNNms step_avg:...
    Returns list of (step, val_loss, train_time_ms).
    """
    all_points = []
    with open(filepath, errors='replace') as f:
        for line in f:
            m = re.match(
                r'^step:(\d+)/\d+\s+val_loss:([0-9.]+)\s+train_time:([0-9.]+)ms',
                line
            )
            if m:
                step = int(m.group(1))
                val_loss = float(m.group(2))
                train_time_ms = float(m.group(3))
                all_points.append((step, val_loss, train_time_ms))
    return all_points


def parse_legacy_log_with_time(filepath: Path):
    """
    Legacy format (AdamW/llmc) doesn't have train_time.
    Returns empty list.
    """
    return []


def split_runs_3(points):
    """Split list of (step, val_loss, train_time_ms) into individual runs at step 0."""
    if not points:
        return []
    runs = []
    current = []
    for pt in points:
        if pt[0] == 0 and current:
            runs.append(current)
            current = []
        current.append(pt)
    if current:
        runs.append(current)
    return runs


def compute_step_epiplexity(points):
    """S_step = ∫[L(step) - L_final] d(step)"""
    if len(points) < 3:
        return None, None, None
    steps = np.array([p[0] for p in points])
    losses = np.array([p[1] for p in points])
    final_loss = losses[-1]
    excess = losses - final_loss
    area = np.trapezoid(excess, steps)
    return float(area), float(final_loss), len(points)


def compute_time_epiplexity(points):
    """S_time = ∫[L(t) - L_final] dt, t in seconds"""
    if len(points) < 3:
        return None, None, None
    times_s = np.array([p[2] / 1000.0 for p in points])  # ms → seconds
    losses = np.array([p[1] for p in points])
    final_loss = losses[-1]
    excess = losses - final_loss
    area = np.trapezoid(excess, times_s)
    return float(area), float(final_loss), len(points)


# ─── Main collection ────────────────────────────────────────────────────────

def collect_all_submissions():
    submissions = []

    for subdir in sorted(TRACK1_DIR.iterdir()):
        if not subdir.is_dir():
            continue

        dirname = subdir.name
        date_str = parse_date(dirname)
        label = extract_label(dirname)
        category = classify_submission(dirname)

        # Collect all log files
        log_files = []
        for ext in ('**/*.txt', '**/*.log'):
            log_files.extend(sorted(subdir.glob(ext)))
        log_files = [f for f in log_files
                     if f.name.lower() != 'readme.md'
                     and not f.name.endswith('.py')]

        all_runs = []
        for lf in log_files:
            points = parse_modern_log_with_time(lf)
            if not points:
                continue

            runs = split_runs_3(points)
            for run_points in runs:
                step_epi, final_loss, n = compute_step_epiplexity(run_points)
                time_epi, _, _ = compute_time_epiplexity(run_points)

                if step_epi is not None and time_epi is not None:
                    total_time_s = run_points[-1][2] / 1000.0
                    all_runs.append({
                        'file': str(lf.relative_to(TRACK1_DIR)),
                        'step_epiplexity': step_epi,
                        'time_epiplexity': time_epi,
                        'final_loss': final_loss,
                        'n_val_points': n,
                        'total_steps': run_points[-1][0],
                        'total_time_s': total_time_s,
                        'points': run_points,
                    })

        if not all_runs:
            continue

        step_epis = [r['step_epiplexity'] for r in all_runs]
        time_epis = [r['time_epiplexity'] for r in all_runs]
        finals = [r['final_loss'] for r in all_runs]
        times = [r['total_time_s'] for r in all_runs]

        submissions.append({
            'dirname': dirname,
            'date': date_str,
            'label': label,
            'category': category,
            'n_runs': len(all_runs),
            'step_epi_mean': float(np.mean(step_epis)),
            'step_epi_std': float(np.std(step_epis)) if len(step_epis) > 1 else 0.0,
            'time_epi_mean': float(np.mean(time_epis)),
            'time_epi_std': float(np.std(time_epis)) if len(time_epis) > 1 else 0.0,
            'final_loss_mean': float(np.mean(finals)),
            'total_steps': int(np.mean([r['total_steps'] for r in all_runs])),
            'total_time_s_mean': float(np.mean(times)),
            'total_time_s_std': float(np.std(times)) if len(times) > 1 else 0.0,
            'runs': all_runs,
        })

    return submissions


# ─── Plotting ────────────────────────────────────────────────────────────────

CATEGORY_COLORS = {
    'Architecture': '#2196F3',
    'Optimization': '#FF9800',
    'Engineering': '#9E9E9E',
}

CATEGORY_MARKERS = {
    'Architecture': 'o',
    'Optimization': 's',
    'Engineering': '^',
}


def plot_1_dual_timeline(submissions):
    """Plot 1: Step-based vs Time-based epiplexity timelines (dual y-axes or two subplots)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True)

    for cat in ['Architecture', 'Optimization', 'Engineering']:
        subs = [s for s in submissions if s['category'] == cat]
        if not subs:
            continue
        dates = [datetime.strptime(s['date'], '%Y-%m-%d') for s in subs]
        color = CATEGORY_COLORS[cat]
        marker = CATEGORY_MARKERS[cat]

        # Step-based
        means = [s['step_epi_mean'] for s in subs]
        stds = [s['step_epi_std'] for s in subs]
        ax1.errorbar(dates, means, yerr=stds, fmt=marker, color=color,
                     label=cat, markersize=6, capsize=3, alpha=0.8)

        # Time-based
        means = [s['time_epi_mean'] for s in subs]
        stds = [s['time_epi_std'] for s in subs]
        ax2.errorbar(dates, means, yerr=stds, fmt=marker, color=color,
                     label=cat, markersize=6, capsize=3, alpha=0.8)

    ax1.set_ylabel('Step-based Epiplexity\n∫[L(step) - L_final] d(step)', fontsize=12)
    ax1.set_title('Step-based vs Time-based Epiplexity across NanoGPT Speedrun', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel('Time-based Epiplexity (loss·seconds)\n∫[L(t) - L_final] dt', fontsize=12)
    ax2.set_xlabel('Submission Date', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    outpath = FIGURES_DIR / 'wallclock_dual_timeline.png'
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved {outpath}")


def plot_2_step_vs_time_scatter(submissions):
    """Plot 2: Scatter plot of step-based vs time-based epiplexity, color by category."""
    fig, ax = plt.subplots(figsize=(12, 10))

    for cat in ['Architecture', 'Optimization', 'Engineering']:
        subs = [s for s in submissions if s['category'] == cat]
        if not subs:
            continue
        x = [s['step_epi_mean'] for s in subs]
        y = [s['time_epi_mean'] for s in subs]
        labels_list = [s['label'] for s in subs]
        color = CATEGORY_COLORS[cat]
        marker = CATEGORY_MARKERS[cat]
        ax.scatter(x, y, c=color, marker=marker, s=60, label=cat, alpha=0.7, edgecolors='k', linewidths=0.5)

        # Annotate interesting outliers
        for xi, yi, lab in zip(x, y, labels_list):
            # Annotate points that deviate from the trend
            if xi > 3000 or yi > 1000 or lab in ['SOAP', 'Muon', 'ModernArch', 'ValueEmbed',
                                                    'FlexAttention', 'PairedHeadMuon',
                                                    'ScaleUp1B', '50Bruns', 'DistributedMuon',
                                                    'TritonMuon', 'FA3']:
                ax.annotate(lab, (xi, yi), fontsize=7, alpha=0.8,
                           xytext=(5, 5), textcoords='offset points')

    ax.set_xlabel('Step-based Epiplexity ∫[L - L_final] d(step)', fontsize=12)
    ax.set_ylabel('Time-based Epiplexity ∫[L - L_final] dt (loss·seconds)', fontsize=12)
    ax.set_title('Step-based vs Time-based Epiplexity', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add correlation line
    all_x = [s['step_epi_mean'] for s in submissions]
    all_y = [s['time_epi_mean'] for s in submissions]
    r, p = stats.pearsonr(all_x, all_y)
    slope, intercept = np.polyfit(all_x, all_y, 1)
    x_fit = np.linspace(min(all_x), max(all_x), 100)
    ax.plot(x_fit, slope * x_fit + intercept, 'k--', alpha=0.4, linewidth=1)
    ax.text(0.05, 0.95, f'Pearson r = {r:.3f} (p = {p:.2e})',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    outpath = FIGURES_DIR / 'step_vs_time_epiplexity.png'
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved {outpath}")
    return r, p


def plot_3_category_comparison(submissions):
    """Plot 3: Box plots comparing step-based vs time-based by category."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    categories = ['Architecture', 'Optimization', 'Engineering']

    # Step-based
    data_step = {cat: [s['step_epi_mean'] for s in submissions if s['category'] == cat]
                 for cat in categories}
    active = [c for c in categories if data_step[c]]
    bp1 = ax1.boxplot([data_step[c] for c in active],
                      tick_labels=active, patch_artist=True, notch=True)
    for i, patch in enumerate(bp1['boxes']):
        patch.set_facecolor(CATEGORY_COLORS[active[i]])
        patch.set_alpha(0.6)
    ax1.set_ylabel('Step-based Epiplexity', fontsize=12)
    ax1.set_title('Step-based Epiplexity by Category', fontsize=13)
    ax1.grid(True, alpha=0.3, axis='y')

    # Time-based
    data_time = {cat: [s['time_epi_mean'] for s in submissions if s['category'] == cat]
                 for cat in categories}
    bp2 = ax2.boxplot([data_time[c] for c in active],
                      tick_labels=active, patch_artist=True, notch=True)
    for i, patch in enumerate(bp2['boxes']):
        patch.set_facecolor(CATEGORY_COLORS[active[i]])
        patch.set_alpha(0.6)
    ax2.set_ylabel('Time-based Epiplexity (loss·seconds)', fontsize=12)
    ax2.set_title('Time-based Epiplexity by Category', fontsize=13)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    outpath = FIGURES_DIR / 'wallclock_category_comparison.png'
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved {outpath}")

    # Print stats
    print("\n  Step-based Epiplexity by Category:")
    for cat in categories:
        vals = data_step[cat]
        if vals:
            print(f"    {cat:15s}: n={len(vals):3d}, mean={np.mean(vals):8.1f}, "
                  f"median={np.median(vals):8.1f}, std={np.std(vals):8.1f}")
    print("\n  Time-based Epiplexity by Category:")
    for cat in categories:
        vals = data_time[cat]
        if vals:
            print(f"    {cat:15s}: n={len(vals):3d}, mean={np.mean(vals):8.1f}, "
                  f"median={np.median(vals):8.1f}, std={np.std(vals):8.1f}")


def plot_4_ratio_analysis(submissions):
    """
    Plot 4: The ratio time_epi / step_epi ≈ average time per step.
    This ratio reveals which submissions are "slow per step" vs "fast per step".
    Engineering submissions should have lower ratios (faster per step).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))

    ratios_by_cat = defaultdict(list)
    all_ratios = []
    all_labels = []
    all_dates = []
    all_cats = []

    for s in submissions:
        ratio = s['time_epi_mean'] / s['step_epi_mean'] if s['step_epi_mean'] > 0 else 0
        s['epi_ratio'] = ratio
        ratios_by_cat[s['category']].append(ratio)
        all_ratios.append(ratio)
        all_labels.append(s['label'])
        all_dates.append(datetime.strptime(s['date'], '%Y-%m-%d'))
        all_cats.append(s['category'])

    # Timeline of ratio
    for cat in ['Architecture', 'Optimization', 'Engineering']:
        subs = [s for s in submissions if s['category'] == cat]
        if not subs:
            continue
        dates = [datetime.strptime(s['date'], '%Y-%m-%d') for s in subs]
        ratios = [s['epi_ratio'] for s in subs]
        ax1.scatter(dates, ratios, c=CATEGORY_COLORS[cat], marker=CATEGORY_MARKERS[cat],
                   s=60, label=cat, alpha=0.7, edgecolors='k', linewidths=0.5)

    ax1.set_ylabel('Epiplexity Ratio (time / step) ≈ avg seconds/step', fontsize=12)
    ax1.set_title('Time/Step Epiplexity Ratio over Speedrun History\n'
                  '(Lower = faster per step = engineering improvement)', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Annotate notable ones
    for s in submissions:
        if s['label'] in ['SOAP', 'DistributedMuon', 'FlexAttention', 'TritonMuon',
                          'FA3', 'PairedHeadMuon', 'ModernArch', 'Muon',
                          'FusedLinearReLUSquare']:
            date = datetime.strptime(s['date'], '%Y-%m-%d')
            ax1.annotate(s['label'], (date, s['epi_ratio']), fontsize=7,
                        xytext=(5, 5), textcoords='offset points', alpha=0.8)

    # Box plot of ratio by category
    active_cats = [c for c in ['Architecture', 'Optimization', 'Engineering'] if ratios_by_cat[c]]
    bp = ax2.boxplot([ratios_by_cat[c] for c in active_cats],
                     tick_labels=active_cats, patch_artist=True, notch=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(CATEGORY_COLORS[active_cats[i]])
        patch.set_alpha(0.6)
    ax2.set_ylabel('Epiplexity Ratio (time / step)', fontsize=12)
    ax2.set_title('Time/Step Ratio by Category\n'
                  '(Engineering should be lower if it makes steps faster)', fontsize=13)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    outpath = FIGURES_DIR / 'wallclock_ratio_analysis.png'
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved {outpath}")

    # Stats
    print("\n  Epiplexity Ratio (time/step) by Category:")
    for cat in active_cats:
        vals = ratios_by_cat[cat]
        print(f"    {cat:15s}: n={len(vals):3d}, mean={np.mean(vals):8.3f}, "
              f"median={np.median(vals):8.3f}, std={np.std(vals):8.3f}")


def plot_5_loss_curves_time(submissions):
    """Plot 5: Representative loss curves with time (seconds) as x-axis vs step as x-axis."""
    representative = ['SOAP', 'ModernArch', 'ValueEmbed', 'TritonMuon',
                      'FA3', 'PairedHeadMuon', 'DistributedMuon', 'FlexAttention']
    by_label = {s['label']: s for s in submissions}

    picks = []
    for key in representative:
        if key in by_label:
            picks.append((key, by_label[key]))
    if len(picks) < 4:
        for s in submissions[:6]:
            if s['label'] not in representative:
                picks.append((s['label'], s))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    cmap = plt.cm.tab10

    for i, (display, sub) in enumerate(picks):
        run = sub['runs'][0]
        pts = run['points']
        steps = [p[0] for p in pts]
        losses = [p[1] for p in pts]
        times_s = [p[2] / 1000.0 for p in pts]
        final = run['final_loss']
        color = cmap(i)

        # Step-based
        ax1.plot(steps, losses, color=color, linewidth=1.5,
                label=f"{display}", alpha=0.9)
        ax1.fill_between(steps, losses, final, color=color, alpha=0.1)

        # Time-based
        ax2.plot(times_s, losses, color=color, linewidth=1.5,
                label=f"{display} ({times_s[-1]:.0f}s)", alpha=0.9)
        ax2.fill_between(times_s, losses, final, color=color, alpha=0.1)

    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Validation Loss', fontsize=12)
    ax1.set_title('Loss Curves (Step-based)\nShaded = Step Epiplexity', fontsize=13)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=2.5)

    ax2.set_xlabel('Wall Clock Time (seconds)', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Loss Curves (Time-based)\nShaded = Time Epiplexity', fontsize=13)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=2.5)

    plt.tight_layout()
    outpath = FIGURES_DIR / 'wallclock_loss_curves.png'
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved {outpath}")


def plot_6_delta_analysis(submissions):
    """
    Plot 6: For consecutive submissions, compute Δ(step_epi) vs Δ(time_epi).
    When they diverge, it reveals whether a submission improved by:
    - Making steps faster (time_epi drops more than step_epi) → Engineering
    - Making learning more efficient (step_epi drops more than time_epi) → Architecture/Optimization
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # Sort by date
    sorted_subs = sorted(submissions, key=lambda s: s['date'])

    for i in range(1, len(sorted_subs)):
        prev = sorted_subs[i-1]
        curr = sorted_subs[i]

        d_step = curr['step_epi_mean'] - prev['step_epi_mean']
        d_time = curr['time_epi_mean'] - prev['time_epi_mean']

        cat = curr['category']
        color = CATEGORY_COLORS.get(cat, '#333')
        marker = CATEGORY_MARKERS.get(cat, 'o')

        ax.scatter(d_step, d_time, c=color, marker=marker, s=50, alpha=0.7,
                  edgecolors='k', linewidths=0.3)

        # Label transitions with big changes
        if abs(d_step) > 500 or abs(d_time) > 300:
            ax.annotate(f"{curr['label']}", (d_step, d_time),
                       fontsize=7, alpha=0.8,
                       xytext=(5, 5), textcoords='offset points')

    # Add quadrant lines and labels
    ax.axhline(0, color='k', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='k', linewidth=0.5, alpha=0.5)

    # Diagonal: y = x * (mean_time_epi / mean_step_epi) would be "proportional change"
    # Simpler: y = 0 and x = 0 lines divide into quadrants

    ax.text(0.02, 0.02, 'Both decrease\n(overall improvement)',
            transform=ax.transAxes, fontsize=9, alpha=0.6, color='green')
    ax.text(0.02, 0.95, 'Time ↑, Step ↓\n(slower but more efficient)',
            transform=ax.transAxes, fontsize=9, alpha=0.6, color='orange',
            verticalalignment='top')
    ax.text(0.75, 0.02, 'Time ↓, Step ↑\n(faster but less efficient)',
            transform=ax.transAxes, fontsize=9, alpha=0.6, color='purple')
    ax.text(0.75, 0.95, 'Both increase\n(regression)',
            transform=ax.transAxes, fontsize=9, alpha=0.6, color='red',
            verticalalignment='top')

    # Legend
    for cat in ['Architecture', 'Optimization', 'Engineering']:
        ax.scatter([], [], c=CATEGORY_COLORS[cat], marker=CATEGORY_MARKERS[cat],
                  s=60, label=cat)
    ax.legend(fontsize=10, loc='center right')

    ax.set_xlabel('Δ Step-based Epiplexity (from previous submission)', fontsize=12)
    ax.set_ylabel('Δ Time-based Epiplexity (from previous submission)', fontsize=12)
    ax.set_title('Change in Step vs Time Epiplexity Between Consecutive Submissions\n'
                 'Off-diagonal = divergent behavior (pure speed vs pure learning)', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    outpath = FIGURES_DIR / 'wallclock_delta_analysis.png'
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved {outpath}")


def plot_7_normalized_comparison(submissions):
    """
    Plot 7: Normalized epiplexity per step vs per second.
    step_epi / total_steps = mean excess loss per step
    time_epi / total_time  = mean excess loss per second
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for cat in ['Architecture', 'Optimization', 'Engineering']:
        subs = [s for s in submissions if s['category'] == cat]
        if not subs:
            continue
        color = CATEGORY_COLORS[cat]
        marker = CATEGORY_MARKERS[cat]

        # Per-step
        per_step = [s['step_epi_mean'] / s['total_steps'] if s['total_steps'] > 0 else 0 for s in subs]
        # Per-second
        per_sec = [s['time_epi_mean'] / s['total_time_s_mean'] if s['total_time_s_mean'] > 0 else 0 for s in subs]
        dates = [datetime.strptime(s['date'], '%Y-%m-%d') for s in subs]

        ax1.scatter(dates, per_step, c=color, marker=marker, s=50, label=cat, alpha=0.7)
        ax2.scatter(dates, per_sec, c=color, marker=marker, s=50, label=cat, alpha=0.7)

    ax1.set_ylabel('Mean Excess Loss per Step', fontsize=12)
    ax1.set_title('Step-Normalized Epiplexity\n(learning inefficiency per step)', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax2.set_ylabel('Mean Excess Loss per Second', fontsize=12)
    ax2.set_title('Time-Normalized Epiplexity\n(learning inefficiency per second)', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    outpath = FIGURES_DIR / 'wallclock_normalized.png'
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved {outpath}")


def plot_8_time_epi_timeline_annotated(submissions):
    """
    Plot 8: Time-based epiplexity timeline with key annotations.
    This is the "money plot" — showing time-based epi is more meaningful for Track 1.
    """
    fig, ax = plt.subplots(figsize=(18, 8))

    for cat in ['Architecture', 'Optimization', 'Engineering']:
        subs = [s for s in submissions if s['category'] == cat]
        if not subs:
            continue
        dates = [datetime.strptime(s['date'], '%Y-%m-%d') for s in subs]
        means = [s['time_epi_mean'] for s in subs]
        color = CATEGORY_COLORS[cat]
        marker = CATEGORY_MARKERS[cat]
        ax.scatter(dates, means, c=color, marker=marker, s=60, label=cat,
                  alpha=0.7, edgecolors='k', linewidths=0.5)

    # Annotate key submissions
    annotations = {
        'SOAP': 'SOAP\n(first modern optimizer)',
        'ModernArch': 'ModernArch\n(architecture overhaul)',
        'FlexAttention': 'FlexAttention\n(kernel speedup)',
        'ValueEmbed': 'ValueEmbed\n(arch innovation)',
        'DistributedMuon': 'DistributedMuon',
        'TritonMuon': 'TritonMuon\n(kernel speedup)',
        'FA3': 'FA3\n(FlashAttention 3)',
        'PairedHeadMuon': 'PairedHeadMuon\n(latest)',
        '50Bruns': '50Bruns\n(95K steps!)',
        'ScaleUp1B': 'ScaleUp1B',
    }

    for s in submissions:
        if s['label'] in annotations:
            date = datetime.strptime(s['date'], '%Y-%m-%d')
            ax.annotate(annotations[s['label']], (date, s['time_epi_mean']),
                       fontsize=7, alpha=0.8,
                       xytext=(10, 10), textcoords='offset points',
                       arrowprops=dict(arrowstyle='->', alpha=0.5, lw=0.8))

    ax.set_xlabel('Submission Date', fontsize=12)
    ax.set_ylabel('Time-based Epiplexity (loss·seconds)\n∫[L(t) - L_final] dt', fontsize=12)
    ax.set_title('Time-based Epiplexity: Measuring "Wasted Loss·Time" in NanoGPT Speedrun\n'
                 'Lower = faster convergence in wall clock time', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    outpath = FIGURES_DIR / 'wallclock_time_epi_annotated.png'
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved {outpath}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"Parsing Track 1 logs from: {TRACK1_DIR}")
    print(f"Extracting (step, val_loss, train_time_ms) triplets\n")

    submissions = collect_all_submissions()
    print(f"\nParsed {len(submissions)} submissions with train_time data\n")

    # ── Summary table ──
    print(f"{'#':>3} {'Date':>10} {'Label':<35} {'Cat':<6} "
          f"{'Runs':>4} {'Steps':>6} {'Time(s)':>8} "
          f"{'StepEpi':>10} {'TimeEpi':>10} {'Ratio':>7}")
    print("─" * 115)
    for i, s in enumerate(submissions, 1):
        ratio = s['time_epi_mean'] / s['step_epi_mean'] if s['step_epi_mean'] > 0 else 0
        cat_short = s['category'][:4]
        print(f"{i:>3} {s['date']:>10} {s['label']:<35} {cat_short:<6} "
              f"{s['n_runs']:>4} {s['total_steps']:>6} {s['total_time_s_mean']:>8.1f} "
              f"{s['step_epi_mean']:>10.1f} {s['time_epi_mean']:>10.1f} "
              f"{ratio:>7.3f}")

    # ── Plots ──
    print("\nGenerating plots...")
    plot_1_dual_timeline(submissions)
    r, p = plot_2_step_vs_time_scatter(submissions)
    print(f"  Pearson correlation (step vs time epi): r={r:.4f}, p={p:.2e}")
    plot_3_category_comparison(submissions)
    plot_4_ratio_analysis(submissions)
    plot_5_loss_curves_time(submissions)
    plot_6_delta_analysis(submissions)
    plot_7_normalized_comparison(submissions)
    plot_8_time_epi_timeline_annotated(submissions)

    # ── Key analysis: Does time-based epi better separate categories? ──
    print("\n\n═══ KEY ANALYSIS: Category Separation ═══")

    for metric_name, metric_key in [('Step-based', 'step_epi_mean'), ('Time-based', 'time_epi_mean')]:
        print(f"\n  {metric_name} Epiplexity:")
        groups = {}
        for cat in ['Architecture', 'Optimization', 'Engineering']:
            groups[cat] = [s[metric_key] for s in submissions if s['category'] == cat]
            vals = groups[cat]
            if vals:
                print(f"    {cat:15s}: n={len(vals):3d}, mean={np.mean(vals):10.1f}, "
                      f"median={np.median(vals):10.1f}, std={np.std(vals):10.1f}")

        # Kruskal-Wallis test (non-parametric ANOVA)
        active_groups = [groups[c] for c in groups if len(groups[c]) >= 3]
        if len(active_groups) >= 2:
            H, p_kw = stats.kruskal(*active_groups)
            print(f"    Kruskal-Wallis H={H:.3f}, p={p_kw:.4f}")

        # Pairwise Mann-Whitney U tests
        cats = [c for c in ['Architecture', 'Optimization', 'Engineering'] if len(groups[c]) >= 3]
        for i in range(len(cats)):
            for j in range(i+1, len(cats)):
                u, p_u = stats.mannwhitneyu(groups[cats[i]], groups[cats[j]], alternative='two-sided')
                print(f"    {cats[i]} vs {cats[j]}: U={u:.0f}, p={p_u:.4f}")

    # ── Save JSON ──
    output = {
        'meta': {
            'description': 'Wall-clock time-based epiplexity analysis for Track 1',
            'n_submissions': len(submissions),
            'excluded': ['AdamW (no train_time)', 'llmc (no train_time)'],
        },
        'submissions': [
            {k: v for k, v in s.items() if k != 'runs'}
            for s in submissions
        ],
    }
    outpath = Path(__file__).parent / 'wallclock_epiplexity.json'
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nSaved JSON to {outpath}")
