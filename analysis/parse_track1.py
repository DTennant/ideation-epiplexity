"""
Parse modded-nanogpt Track 1 (main speedrun) log files
and compute epiplexity (area under loss curve above final loss)
for each submission.

Epiplexity = integral of [L(t) - L_final] dt  (trapezoidal rule over steps)

Supports two log formats:
  - Modern: step:N/TOTAL val_loss:X.XXXX train_time:... step_avg:...
  - Legacy (llmc/AdamW): s:N tel:X.XXXX  (val loss logged as 'tel')
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

TRACK1_DIR = Path(os.environ.get(
    "TRACK1_DIR",
    os.path.expanduser("~/modded-nanogpt-ref/records/track_1_short")
))

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ─── Category classification ───────────────────────────────────────────────────

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
    """Classify a submission name into Architecture / Optimization / Engineering."""
    # Strip date prefix to get the label
    # Handle both 2024-10-14_ModernArch and 2026-01-26-UntieValueEmbeddings formats
    parts = re.split(r'^\d{4}-\d{2}-\d{2}[-_]', name)
    label = parts[-1] if len(parts) > 1 else name

    # Direct keyword match
    if label in ARCHITECTURE_KEYWORDS:
        return 'Architecture'
    if label in OPTIMIZATION_KEYWORDS:
        return 'Optimization'
    if label in ENGINEERING_KEYWORDS:
        return 'Engineering'

    # Fuzzy: check if any keyword is a substring
    for kw in ARCHITECTURE_KEYWORDS:
        if kw.lower() in label.lower():
            return 'Architecture'
    for kw in OPTIMIZATION_KEYWORDS:
        if kw.lower() in label.lower():
            return 'Optimization'
    for kw in ENGINEERING_KEYWORDS:
        if kw.lower() in label.lower():
            return 'Engineering'

    return 'Optimization'  # default fallback


def parse_date(dirname: str) -> str:
    """Extract date string from directory name like '2024-10-14_ModernArch'."""
    m = re.match(r'(\d{4}-\d{2}-\d{2})', dirname)
    return m.group(1) if m else '1970-01-01'


def extract_label(dirname: str) -> str:
    """Extract human-readable label from directory name."""
    parts = re.split(r'^\d{4}-\d{2}-\d{2}[-_]', dirname)
    return parts[-1] if len(parts) > 1 else dirname


# ─── Log parsing ───────────────────────────────────────────────────────────────

def parse_modern_log(filepath: Path):
    """Parse modern format: step:N/TOTAL val_loss:X.XXXX ..."""
    all_points = []
    with open(filepath) as f:
        for line in f:
            m = re.match(r'^step:(\d+)/\d+\s+val_loss:([0-9.]+)', line)
            if m:
                step = int(m.group(1))
                val_loss = float(m.group(2))
                all_points.append((step, val_loss))
    return all_points


def parse_legacy_log(filepath: Path):
    """Parse legacy format: s:N tel:X.XXXX (llmc/AdamW style)."""
    all_points = []
    with open(filepath) as f:
        for line in f:
            m = re.match(r'^s:(\d+)\s+tel:([0-9.]+)', line)
            if m:
                step = int(m.group(1))
                val_loss = float(m.group(2))
                all_points.append((step, val_loss))
    return all_points


def split_runs(points):
    """Split a list of (step, val_loss) into individual runs at step 0 resets."""
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


def compute_epiplexity(points):
    """
    S = integral of [L(t) - L_final] dt using trapezoidal rule.
    Returns (epiplexity, final_loss, n_points).
    """
    if len(points) < 3:
        return None, None, None

    steps = np.array([p[0] for p in points])
    losses = np.array([p[1] for p in points])
    final_loss = losses[-1]

    excess = losses - final_loss
    area = np.trapezoid(excess, steps)

    return float(area), float(final_loss), len(points)


# ─── Main collection ──────────────────────────────────────────────────────────

def collect_all_submissions():
    """Walk Track 1 directory and parse all submissions."""
    submissions = []

    for subdir in sorted(TRACK1_DIR.iterdir()):
        if not subdir.is_dir():
            continue

        dirname = subdir.name
        date_str = parse_date(dirname)
        label = extract_label(dirname)
        category = classify_submission(dirname)

        # Collect all log files recursively (support .txt, .log in any subdirectory)
        log_files = []
        for ext in ('**/*.txt', '**/*.log'):
            log_files.extend(sorted(subdir.glob(ext)))

        # Filter out non-log files (READMEs, code files, .py, etc.)
        log_files = [f for f in log_files
                     if f.name.lower() != 'readme.md'
                     and not f.name.endswith('.py')]

        all_runs = []
        for lf in log_files:
            # Try modern format first
            points = parse_modern_log(lf)
            if not points:
                # Try legacy format
                points = parse_legacy_log(lf)
            if not points:
                continue

            runs = split_runs(points)
            for run_points in runs:
                epi, final, n = compute_epiplexity(run_points)
                if epi is not None:
                    all_runs.append({
                        'file': str(lf.relative_to(TRACK1_DIR)),
                        'epiplexity': epi,
                        'final_loss': final,
                        'n_val_points': n,
                        'total_steps': run_points[-1][0],
                        'points': run_points,
                    })

        if not all_runs:
            print(f"  SKIP {dirname}: no valid val_loss data")
            continue

        # Aggregate
        epis = [r['epiplexity'] for r in all_runs]
        finals = [r['final_loss'] for r in all_runs]

        submissions.append({
            'dirname': dirname,
            'date': date_str,
            'label': label,
            'category': category,
            'n_runs': len(all_runs),
            'epiplexity_mean': float(np.mean(epis)),
            'epiplexity_std': float(np.std(epis)) if len(epis) > 1 else 0.0,
            'epiplexity_min': float(np.min(epis)),
            'epiplexity_max': float(np.max(epis)),
            'final_loss_mean': float(np.mean(finals)),
            'final_loss_std': float(np.std(finals)) if len(finals) > 1 else 0.0,
            'total_steps': int(np.mean([r['total_steps'] for r in all_runs])),
            'runs': all_runs,
        })

    return submissions


# ─── Plotting ──────────────────────────────────────────────────────────────────

CATEGORY_COLORS = {
    'Architecture': '#2196F3',   # blue
    'Optimization': '#FF9800',   # orange
    'Engineering': '#9E9E9E',    # gray
}


def plot_timeline(submissions):
    """Plot A: Epiplexity timeline with color-coded categories."""
    fig, ax = plt.subplots(figsize=(18, 7))

    for cat in ['Architecture', 'Optimization', 'Engineering']:
        subs = [s for s in submissions if s['category'] == cat]
        if not subs:
            continue
        dates = [datetime.strptime(s['date'], '%Y-%m-%d') for s in subs]
        means = [s['epiplexity_mean'] for s in subs]
        stds = [s['epiplexity_std'] for s in subs]
        ax.errorbar(dates, means, yerr=stds, fmt='o', color=CATEGORY_COLORS[cat],
                     label=cat, markersize=6, capsize=3, alpha=0.8)

    ax.set_xlabel('Submission Date', fontsize=12)
    ax.set_ylabel('Mean Epiplexity (step-integrated excess loss)', fontsize=12)
    ax.set_title('Epiplexity across NanoGPT Speedrun History (Track 1)', fontsize=14)
    ax.legend(fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    outpath = FIGURES_DIR / 'epiplexity_timeline.png'
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved {outpath}")


def plot_by_category(submissions):
    """Plot B: Box/violin plot of epiplexity by category."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    categories = ['Architecture', 'Optimization', 'Engineering']
    data = {cat: [s['epiplexity_mean'] for s in submissions if s['category'] == cat]
            for cat in categories}
    colors = [CATEGORY_COLORS[c] for c in categories]

    # Violin plot
    ax = axes[0]
    parts = ax.violinplot([data[c] for c in categories if data[c]],
                          positions=range(len([c for c in categories if data[c]])),
                          showmeans=True, showmedians=True)
    active_cats = [c for c in categories if data[c]]
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(CATEGORY_COLORS[active_cats[i]])
        pc.set_alpha(0.6)
    ax.set_xticks(range(len(active_cats)))
    ax.set_xticklabels(active_cats, fontsize=11)
    ax.set_ylabel('Mean Epiplexity', fontsize=12)
    ax.set_title('Epiplexity Distribution by Category (Violin)', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    # Box plot
    ax = axes[1]
    bp = ax.boxplot([data[c] for c in categories if data[c]],
                    tick_labels=[c for c in categories if data[c]],
                    patch_artist=True, notch=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(CATEGORY_COLORS[active_cats[i]])
        patch.set_alpha(0.6)
    ax.set_ylabel('Mean Epiplexity', fontsize=12)
    ax.set_title('Epiplexity Distribution by Category (Box)', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    # Print stats
    for cat in categories:
        vals = data[cat]
        if vals:
            print(f"  {cat:15s}: n={len(vals):3d}, mean={np.mean(vals):8.1f}, "
                  f"median={np.median(vals):8.1f}, std={np.std(vals):8.1f}")

    plt.tight_layout()
    outpath = FIGURES_DIR / 'epiplexity_by_category.png'
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved {outpath}")


def plot_loss_curves(submissions):
    """Plot C: Overlay loss curves for representative submissions with shaded epiplexity."""
    # Pick ~5 representative submissions across categories and history
    representative_labels = {
        'llmc': 'llmc (baseline)',
        'ModernArch': 'ModernArch',
        'Muon': 'Muon',
        'ValueEmbed': 'ValueEmbed',
        'PairedHeadMuon': 'PairedHeadMuon',
    }

    # Build lookup by label
    by_label = {s['label']: s for s in submissions}
    picks = []
    for key, display in representative_labels.items():
        if key in by_label:
            picks.append((display, by_label[key]))

    # If we didn't find enough, add some more
    if len(picks) < 4:
        for s in submissions:
            if s['label'] not in representative_labels and len(picks) < 5:
                picks.append((s['label'], s))

    fig, ax = plt.subplots(figsize=(14, 8))
    cmap = plt.cm.tab10

    for i, (display, sub) in enumerate(picks):
        # Pick representative run (first one)
        run = sub['runs'][0]
        steps = [p[0] for p in run['points']]
        losses = [p[1] for p in run['points']]
        final = run['final_loss']
        color = cmap(i)

        ax.plot(steps, losses, color=color, linewidth=1.5,
                label=f"{display} (epi={run['epiplexity']:.0f})", alpha=0.9)
        ax.fill_between(steps, losses, final, color=color, alpha=0.15)
        ax.axhline(y=final, color=color, linestyle=':', alpha=0.4, linewidth=0.8)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Loss Curves with Shaded Epiplexity (Representative Submissions)', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=2.5)
    plt.tight_layout()

    outpath = FIGURES_DIR / 'loss_curves_overlay.png'
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved {outpath}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"Parsing Track 1 logs from: {TRACK1_DIR}\n")

    submissions = collect_all_submissions()
    print(f"\nParsed {len(submissions)} submissions total\n")

    # ── Summary table ──
    print(f"{'#':>3} {'Date':>10} {'Label':<35} {'Category':<14} "
          f"{'Runs':>4} {'Steps':>6} {'Epiplexity':>11} {'±':>8} {'Final Loss':>10}")
    print("─" * 115)
    for i, s in enumerate(submissions, 1):
        print(f"{i:>3} {s['date']:>10} {s['label']:<35} {s['category']:<14} "
              f"{s['n_runs']:>4} {s['total_steps']:>6} "
              f"{s['epiplexity_mean']:>11.1f} {s['epiplexity_std']:>8.1f} "
              f"{s['final_loss_mean']:>10.4f}")

    # ── Plots ──
    print("\nGenerating plots...")
    plot_timeline(submissions)
    plot_by_category(submissions)
    plot_loss_curves(submissions)

    # ── Save JSON ──
    output = {
        'submissions': [
            {k: v for k, v in s.items() if k != 'runs'}
            for s in submissions
        ],
        'submissions_with_runs': [
            {
                **{k: v for k, v in s.items() if k != 'runs'},
                'runs': [
                    {k: v for k, v in r.items() if k != 'points'}
                    for r in s['runs']
                ],
            }
            for s in submissions
        ],
    }
    outpath = Path(__file__).parent / 'track1_epiplexity.json'
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nSaved JSON to {outpath}")

    # ── Category summary ──
    print("\n── Category Summary ──")
    for cat in ['Architecture', 'Optimization', 'Engineering']:
        subs = [s for s in submissions if s['category'] == cat]
        if subs:
            epis = [s['epiplexity_mean'] for s in subs]
            print(f"  {cat:15s}: {len(subs):3d} submissions, "
                  f"mean epi = {np.mean(epis):8.1f}, "
                  f"median = {np.median(epis):8.1f}, "
                  f"std = {np.std(epis):8.1f}")
