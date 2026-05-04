#!/usr/bin/env python3
"""
Controlled Epiplexity Analysis for modded-nanogpt Track 1 speedrun records.

Two experiments:
  A) Fixed Step Budget: ∫₀¹⁰⁰⁰ [L(s) - 3.28] ds
  B) Fixed Wall-Clock Budget: ∫₀⁹⁰ˢ [L(t) - 3.28] dt

This controls for the confound that different records have different total
step counts and wall-clock times, making raw epiplexity incomparable.
"""

import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Config ──────────────────────────────────────────────────────────────────
RECORDS_DIR = Path("/home/zhaobc/modded-nanogpt-ref/records/track_1_short")
PROJECT_DIR = Path("/home/zhaobc/ideation-epiplexity")
FIGURES_DIR = PROJECT_DIR / "analysis" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_LOSS = 3.28
STEP_BUDGET = 1000
TIME_BUDGET_S = 90.0

# Directories to skip
SKIP_DIRS = {"2024-10-13_llmc"}  # Different log format, skip per instructions

# ── Category Colors ─────────────────────────────────────────────────────────
CATEGORY_COLORS = {
    "Architecture": "#e74c3c",
    "Optimization": "#3498db",
    "Engineering":  "#2ecc71",
}

# ── Parse log files ─────────────────────────────────────────────────────────

def parse_train_time(s: str) -> float:
    """Parse train_time field, returning milliseconds.
    
    Handles:
      - train_time:12345ms  (integer ms)
      - train_time:12345.6ms (float ms)
      - train_time:123.456s (seconds → convert to ms)
    """
    m = re.match(r'([\d.]+)(ms|s)$', s)
    if not m:
        return float('nan')
    val = float(m.group(1))
    unit = m.group(2)
    if unit == 's':
        val *= 1000.0
    return val


def parse_log_file(filepath: Path):
    """Parse a single log file, extracting (step, val_loss, train_time_ms) tuples.
    
    Only extracts val_loss lines (not train_loss).
    Takes only the first run (stops if step resets to 0 after having seen steps > 0).
    
    Returns list of dicts with keys: step, val_loss, train_time_ms
    """
    records = []
    max_step_seen = -1
    
    with open(filepath, 'r', errors='replace') as f:
        for line in f:
            line = line.strip()
            # Match val_loss lines: step:N/M val_loss:X train_time:Yms step_avg:Zms
            m = re.match(
                r'step:(\d+)/\d+\s+val_loss:([\d.]+)\s+train_time:([\d.]+(?:ms|s))',
                line
            )
            if not m:
                continue
            
            step = int(m.group(1))
            val_loss = float(m.group(2))
            train_time_ms = parse_train_time(m.group(3))
            
            # Detect run reset: if step goes back to 0 after we've seen higher steps
            if step == 0 and max_step_seen > 0:
                break  # Stop at end of first run
            
            max_step_seen = max(max_step_seen, step)
            records.append({
                'step': step,
                'val_loss': val_loss,
                'train_time_ms': train_time_ms,
            })
    
    return records


def parse_adamw_log(filepath: Path):
    """Parse the AdamW-style log format: s:N tel:X (no time info)."""
    records = []
    with open(filepath, 'r', errors='replace') as f:
        for line in f:
            line = line.strip()
            m = re.match(r's:(\d+)\s+tel:([\d.]+)', line)
            if not m:
                continue
            step = int(m.group(1))
            val_loss = float(m.group(2))
            records.append({
                'step': step,
                'val_loss': val_loss,
                'train_time_ms': float('nan'),  # No time info in AdamW format
            })
    return records


def load_all_records():
    """Load val_loss data from all record directories.
    
    Returns dict: dirname -> list of records (averaged across runs if multiple).
    """
    # Load categories from existing analysis
    categories = {}
    epi_json = PROJECT_DIR / "analysis" / "track1_epiplexity.json"
    if epi_json.exists():
        with open(epi_json) as f:
            data = json.load(f)
            for sub in data.get("submissions", []):
                categories[sub["dirname"]] = sub.get("category", "Unknown")
    
    all_data = {}
    
    for d in sorted(RECORDS_DIR.iterdir()):
        if not d.is_dir():
            continue
        dirname = d.name
        if dirname in SKIP_DIRS:
            continue
        
        category = categories.get(dirname, "Unknown")
        
        # Find log files
        # Try .txt files first, then .log files, then search subdirs
        log_files = list(d.glob("*.txt")) + list(d.glob("*.log"))
        if not log_files:
            # Search subdirectories
            log_files = list(d.rglob("*.txt"))
        
        if not log_files:
            print(f"WARNING: No log files found for {dirname}, skipping")
            continue
        
        # Check if this is AdamW format
        is_adamw = dirname == "2024-06-06_AdamW"
        
        # Parse all log files for this record, pick the best one
        # (most val_loss data points from the first non-comparison file)
        best_records = None
        for lf in log_files:
            # Skip comparison files
            if 'comparison' in lf.name.lower():
                continue
            
            if is_adamw:
                recs = parse_adamw_log(lf)
            else:
                recs = parse_log_file(lf)
            
            if recs and (best_records is None or len(recs) > len(best_records)):
                best_records = recs
        
        if not best_records:
            print(f"WARNING: No valid records for {dirname}, skipping")
            continue
        
        all_data[dirname] = {
            'records': best_records,
            'category': category,
            'date': dirname.split('_')[0],
            'label': '_'.join(dirname.split('_')[1:]),
        }
    
    return all_data


# ── Experiment A: Fixed Step Budget ─────────────────────────────────────────

def compute_step_controlled_epiplexity(records, step_budget=STEP_BUDGET, ref_loss=REFERENCE_LOSS):
    """Compute ∫₀ˢᵗᵉᵖ_ᵇᵘᵈᵍᵉᵗ [L(s) - ref_loss] ds using trapezoidal rule.
    
    Interpolates to a fine grid (step=1) to avoid bias from different eval intervals
    (some records eval every 125 steps, others every 250).
    Returns None if max step < step_budget.
    """
    # Get all points, sorted by step
    points = [(r['step'], r['val_loss']) for r in records]
    if not points:
        return None
    points.sort(key=lambda x: x[0])
    
    steps = np.array([p[0] for p in points])
    losses = np.array([p[1] for p in points])
    
    # Check if we have data reaching up to step_budget
    max_step = steps[-1]
    if max_step < step_budget:
        return None
    
    # Interpolate to a fine grid to avoid eval-interval bias
    # Use linear interpolation between observed points
    fine_steps = np.arange(0, step_budget + 1, 1)
    fine_losses = np.interp(fine_steps, steps, losses)
    
    # Compute area: ∫ [L(s) - ref_loss] ds
    excess = fine_losses - ref_loss
    area = np.trapezoid(excess, fine_steps)
    
    return area


# ── Experiment B: Fixed Wall-Clock Budget ───────────────────────────────────

def compute_time_controlled_epiplexity(records, time_budget_s=TIME_BUDGET_S, ref_loss=REFERENCE_LOSS):
    """Compute ∫₀ᵗⁱᵐᵉ_ᵇᵘᵈᵍᵉᵗ [L(t) - ref_loss] dt using trapezoidal rule.
    
    Uses (train_time_ms, val_loss) pairs, converting train_time to seconds.
    Returns None if no time data available.
    """
    time_budget_ms = time_budget_s * 1000.0
    
    # Filter records with valid time data
    points = []
    for r in records:
        if np.isnan(r['train_time_ms']):
            continue
        points.append((r['train_time_ms'], r['val_loss']))
    
    if len(points) < 2:
        return None
    
    # Sort by time
    points.sort(key=lambda x: x[0])
    
    times_ms = np.array([p[0] for p in points])
    losses = np.array([p[1] for p in points])
    
    # Convert to seconds for integration
    times_s = times_ms / 1000.0
    
    # Clip to time budget
    if times_s[-1] < time_budget_s:
        # Record didn't reach time budget — use all available data
        # This means the record finished before the budget, which is good
        # The loss at end is likely at or below target
        pass
    elif times_s[0] > time_budget_s:
        return None
    else:
        # Interpolate to time_budget
        idx = np.searchsorted(times_s, time_budget_s)
        if idx < len(times_s) and abs(times_s[idx] - time_budget_s) < 0.001:
            # Exact match
            mask = times_s <= time_budget_s + 0.001
            times_s = times_s[mask]
            losses = losses[mask]
        elif idx > 0 and idx < len(times_s):
            t_before = times_s[idx - 1]
            t_after = times_s[idx]
            l_before = losses[idx - 1]
            l_after = losses[idx]
            frac = (time_budget_s - t_before) / (t_after - t_before)
            l_at_budget = l_before + frac * (l_after - l_before)
            
            times_s = np.append(times_s[:idx], time_budget_s)
            losses = np.append(losses[:idx], l_at_budget)
        else:
            # idx is 0 or beyond array
            times_s = times_s[:idx]
            losses = losses[:idx]
    
    if len(times_s) < 2:
        return None
    
    # Compute area: ∫ [L(t) - ref_loss] dt
    excess = losses - ref_loss
    area = np.trapezoid(excess, times_s)
    
    return area


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_step_comparison(results, output_path):
    """Bar chart of step-controlled epiplexity, sorted by date, colored by category."""
    fig, ax = plt.subplots(figsize=(20, 8))
    
    # Sort by date
    sorted_results = sorted(results, key=lambda r: r['date'])
    
    names = []
    values = []
    colors = []
    for r in sorted_results:
        if r['step_epi'] is None:
            continue
        # Short label: date + name
        label = f"{r['date']}\n{r['label']}"
        names.append(label)
        values.append(r['step_epi'])
        colors.append(CATEGORY_COLORS.get(r['category'], '#95a5a6'))
    
    bars = ax.bar(range(len(names)), values, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6, ha='center')
    ax.set_ylabel(f'Controlled Epiplexity (step budget = {STEP_BUDGET})', fontsize=12)
    ax.set_title(f'Experiment A: Fixed Step Budget — ∫₀^{STEP_BUDGET} [L(s) - {REFERENCE_LOSS}] ds', fontsize=14)
    
    # Legend
    legend_elements = [Patch(facecolor=c, label=cat, alpha=0.85) 
                       for cat, c in CATEGORY_COLORS.items()]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(-0.5, len(names) - 0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_time_comparison(results, output_path):
    """Bar chart of time-controlled epiplexity, sorted by date, colored by category."""
    fig, ax = plt.subplots(figsize=(20, 8))
    
    sorted_results = sorted(results, key=lambda r: r['date'])
    
    names = []
    values = []
    colors = []
    for r in sorted_results:
        if r['time_epi'] is None:
            continue
        label = f"{r['date']}\n{r['label']}"
        names.append(label)
        values.append(r['time_epi'])
        colors.append(CATEGORY_COLORS.get(r['category'], '#95a5a6'))
    
    bars = ax.bar(range(len(names)), values, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6, ha='center')
    ax.set_ylabel(f'Controlled Epiplexity (time budget = {TIME_BUDGET_S}s)', fontsize=12)
    ax.set_title(f'Experiment B: Fixed Wall-Clock Budget — ∫₀^{TIME_BUDGET_S}s [L(t) - {REFERENCE_LOSS}] dt', fontsize=14)
    
    legend_elements = [Patch(facecolor=c, label=cat, alpha=0.85) 
                       for cat, c in CATEGORY_COLORS.items()]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(-0.5, len(names) - 0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_2d_scatter(results, output_path):
    """Scatter: x = step-controlled epi, y = time-controlled epi, colored by category."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for cat, color in CATEGORY_COLORS.items():
        xs = []
        ys = []
        labels = []
        for r in results:
            if r['category'] != cat:
                continue
            if r['step_epi'] is None or r['time_epi'] is None:
                continue
            xs.append(r['step_epi'])
            ys.append(r['time_epi'])
            labels.append(r['label'])
        
        ax.scatter(xs, ys, c=color, label=cat, s=60, alpha=0.8, edgecolors='white', linewidth=0.5)
        
        # Annotate points
        for x, y, lab in zip(xs, ys, labels):
            ax.annotate(lab, (x, y), fontsize=5, alpha=0.7,
                       xytext=(3, 3), textcoords='offset points')
    
    ax.set_xlabel(f'Step-Controlled Epiplexity (∫₀^{STEP_BUDGET} [L(s)-{REFERENCE_LOSS}] ds)', fontsize=11)
    ax.set_ylabel(f'Time-Controlled Epiplexity (∫₀^{TIME_BUDGET_S}s [L(t)-{REFERENCE_LOSS}] dt)', fontsize=11)
    ax.set_title('2D Scatter: Step vs Time Controlled Epiplexity', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_category_boxplot(results, output_path):
    """Side-by-side boxplots per category for step-epi and time-epi."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    cats_order = ["Architecture", "Optimization", "Engineering"]
    
    # Step-controlled
    ax = axes[0]
    data_step = []
    labels_step = []
    colors_step = []
    for cat in cats_order:
        vals = [r['step_epi'] for r in results if r['category'] == cat and r['step_epi'] is not None]
        if vals:
            data_step.append(vals)
            labels_step.append(f"{cat}\n(n={len(vals)})")
            colors_step.append(CATEGORY_COLORS[cat])
    
    bp1 = ax.boxplot(data_step, tick_labels=labels_step, patch_artist=True, widths=0.6)
    for patch, color in zip(bp1['boxes'], colors_step):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    # Also plot individual points
    for i, (vals, color) in enumerate(zip(data_step, colors_step)):
        x = np.random.normal(i + 1, 0.04, size=len(vals))
        ax.scatter(x, vals, c=color, s=20, alpha=0.6, edgecolors='white', linewidth=0.3, zorder=3)
    
    ax.set_ylabel(f'Step-Controlled Epiplexity', fontsize=11)
    ax.set_title(f'Experiment A: Step Budget = {STEP_BUDGET}', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Time-controlled
    ax = axes[1]
    data_time = []
    labels_time = []
    colors_time = []
    for cat in cats_order:
        vals = [r['time_epi'] for r in results if r['category'] == cat and r['time_epi'] is not None]
        if vals:
            data_time.append(vals)
            labels_time.append(f"{cat}\n(n={len(vals)})")
            colors_time.append(CATEGORY_COLORS[cat])
    
    bp2 = ax.boxplot(data_time, tick_labels=labels_time, patch_artist=True, widths=0.6)
    for patch, color in zip(bp2['boxes'], colors_time):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for i, (vals, color) in enumerate(zip(data_time, colors_time)):
        x = np.random.normal(i + 1, 0.04, size=len(vals))
        ax.scatter(x, vals, c=color, s=20, alpha=0.6, edgecolors='white', linewidth=0.3, zorder=3)
    
    ax.set_ylabel(f'Time-Controlled Epiplexity', fontsize=11)
    ax.set_title(f'Experiment B: Time Budget = {TIME_BUDGET_S}s', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Controlled Epiplexity by Category', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("Loading records...")
    all_data = load_all_records()
    print(f"  Loaded {len(all_data)} records")
    
    results = []
    
    print(f"\nComputing controlled epiplexity...")
    print(f"  Step budget: {STEP_BUDGET}")
    print(f"  Time budget: {TIME_BUDGET_S}s")
    print(f"  Reference loss: {REFERENCE_LOSS}")
    print()
    
    for dirname in sorted(all_data.keys()):
        info = all_data[dirname]
        records = info['records']
        
        step_epi = compute_step_controlled_epiplexity(records)
        time_epi = compute_time_controlled_epiplexity(records)
        
        results.append({
            'dirname': dirname,
            'date': info['date'],
            'label': info['label'],
            'category': info['category'],
            'step_epi': step_epi,
            'time_epi': time_epi,
            'n_val_points': len(records),
            'max_step': max(r['step'] for r in records) if records else 0,
        })
        
        step_str = f"{step_epi:.1f}" if step_epi is not None else "N/A"
        time_str = f"{time_epi:.1f}" if time_epi is not None else "N/A"
        print(f"  {dirname:50s} {info['category']:15s} step_epi={step_str:>10s}  time_epi={time_str:>10s}")
    
    # Summary stats
    print("\n" + "=" * 80)
    print("SUMMARY BY CATEGORY")
    print("=" * 80)
    
    for cat in ["Architecture", "Optimization", "Engineering"]:
        step_vals = [r['step_epi'] for r in results if r['category'] == cat and r['step_epi'] is not None]
        time_vals = [r['time_epi'] for r in results if r['category'] == cat and r['time_epi'] is not None]
        
        print(f"\n{cat} (n_step={len(step_vals)}, n_time={len(time_vals)}):")
        if step_vals:
            print(f"  Step-controlled:  mean={np.mean(step_vals):.1f}  std={np.std(step_vals):.1f}  "
                  f"min={np.min(step_vals):.1f}  max={np.max(step_vals):.1f}")
        if time_vals:
            print(f"  Time-controlled:  mean={np.mean(time_vals):.1f}  std={np.std(time_vals):.1f}  "
                  f"min={np.min(time_vals):.1f}  max={np.max(time_vals):.1f}")
    
    # Plot
    print("\nGenerating figures...")
    plot_step_comparison(results, FIGURES_DIR / "controlled_step_comparison.png")
    plot_time_comparison(results, FIGURES_DIR / "controlled_time_comparison.png")
    plot_2d_scatter(results, FIGURES_DIR / "controlled_2d_scatter.png")
    plot_category_boxplot(results, FIGURES_DIR / "controlled_category_boxplot.png")
    
    # Save results JSON
    output_json = PROJECT_DIR / "analysis" / "controlled_epiplexity.json"
    with open(output_json, 'w') as f:
        json.dump({
            'config': {
                'step_budget': STEP_BUDGET,
                'time_budget_s': TIME_BUDGET_S,
                'reference_loss': REFERENCE_LOSS,
            },
            'results': results,
        }, f, indent=2)
    print(f"\n  Saved results: {output_json}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
