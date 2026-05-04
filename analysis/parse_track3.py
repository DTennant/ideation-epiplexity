"""
Parse modded-nanogpt Track 3 optimization benchmark logs
and compute epiplexity (area under loss curve above final loss).
"""
import os
import re
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

TRACK3_DIR = Path(os.environ.get(
    "TRACK3_DIR",
    os.path.expanduser("~/modded-nanogpt-ref/records/track_3_optimization/results")
))

# Mapping from directory/file to optimizer name + metadata
# Based on the README table
OPTIMIZER_INFO = {
    # Standalone files (baseline runs)
    "7b8270c5": {"name": "Muon + aux Adam", "entry": 1, "steps": 3600, "lr": 0.02, "wd": 0.01},
    "a63a68d1": {"name": "AdamW", "entry": 2, "steps": 5625, "lr": 0.0015, "wd": 0.1},
    "311d7833": {"name": "Muon + aux Adam (tuned)", "entry": 3, "steps": 3500, "lr": 0.025, "wd": 0.0125},
    "51ece938": {"name": "Muon + aux Adam (v3)", "entry": 6, "steps": 3375, "lr": 0.025, "wd": 0.025},
    "1bd8db7a": {"name": "Muon + aux Adam (v4)", "entry": 12, "steps": 3325, "lr": 0.035, "wd": 0.025},
    # Directories
    "20260430_adamh": {"name": "AdamH", "entry": 4, "steps": 4875},
    "20260430_muonh": {"name": "MuonH", "entry": 5, "steps": 3325},
    "20260430_normuonh": {"name": "NorMuonH", "entry": 8, "steps": 3250},
    "20260501_contra_muon": {"name": "Contra-Muon", "entry": 11, "steps": 3225},
    "20260501_muonsq": {"name": "Muon²", "entry": 7, "steps": 3325},
    "20260501_skylight001": {"name": "NorMuon + u/w-floor", "entry": 9, "steps": 3250},
    "20260503_normuon": {"name": "NorMuon", "entry": 10, "steps": 3250},
}


def parse_log(filepath):
    """Extract (step, val_loss) pairs from a log file.
    
    Some files contain multiple runs concatenated (step resets to 0).
    We split into individual runs and return a list of runs.
    """
    all_points = []
    with open(filepath) as f:
        for line in f:
            m = re.match(r'step:(\d+)/\d+ val_loss:([0-9.]+)', line)
            if m:
                step = int(m.group(1))
                val_loss = float(m.group(2))
                all_points.append((step, val_loss))
    
    if not all_points:
        return []
    
    # Split into individual runs at step 0 resets
    runs = []
    current_run = []
    for pt in all_points:
        if pt[0] == 0 and current_run:
            runs.append(current_run)
            current_run = []
        current_run.append(pt)
    if current_run:
        runs.append(current_run)
    
    return runs


def compute_epiplexity(points):
    """
    Compute epiplexity as area between loss curve and final loss,
    using trapezoidal integration over steps.
    
    S = integral[ L(t) - L(inf) ] dt
    
    where t is measured in steps and L(inf) is the final loss value.
    """
    if len(points) < 2:
        return None, None, None
    
    steps = np.array([p[0] for p in points])
    losses = np.array([p[1] for p in points])
    final_loss = losses[-1]
    
    # Area above final loss using trapezoidal rule
    excess = losses - final_loss
    area = np.trapezoid(excess, steps)
    
    return area, final_loss, len(points)


def _add_runs_from_file(runs, filepath, info, file_label):
    """Parse a log file and add individual runs to the list."""
    parsed_runs = parse_log(filepath)
    for run_points in parsed_runs:
        area, final_loss, n_points = compute_epiplexity(run_points)
        if area is not None:
            runs.append({
                "optimizer": info["name"],
                "entry": info.get("entry", -1),
                "file": file_label,
                "epiplexity": area,
                "final_loss": final_loss,
                "n_val_points": n_points,
                "total_steps": run_points[-1][0] if run_points else 0,
                "points": run_points,
            })


def collect_all_runs():
    """Collect all runs from Track 3 results directory."""
    runs = []
    
    for item in sorted(TRACK3_DIR.iterdir()):
        if item.is_file() and item.suffix == '.txt':
            key = item.stem[:8]
            info = OPTIMIZER_INFO.get(key, {"name": f"Unknown ({key})", "entry": -1})
            _add_runs_from_file(runs, item, info, item.name)
        
        elif item.is_dir():
            key = item.name
            info = OPTIMIZER_INFO.get(key, {"name": f"Unknown ({key})", "entry": -1})
            for logfile in sorted(item.glob("*.txt")):
                _add_runs_from_file(runs, logfile, info, f"{key}/{logfile.name}")
    
    return runs


def aggregate_by_optimizer(runs):
    """Aggregate runs by optimizer, computing mean/std of epiplexity."""
    by_opt = defaultdict(list)
    for r in runs:
        by_opt[r["optimizer"]].append(r)
    
    results = []
    for opt_name, opt_runs in by_opt.items():
        epis = [r["epiplexity"] for r in opt_runs]
        finals = [r["final_loss"] for r in opt_runs]
        steps = [r["total_steps"] for r in opt_runs]
        results.append({
            "optimizer": opt_name,
            "entry": opt_runs[0]["entry"],
            "n_runs": len(opt_runs),
            "epiplexity_mean": np.mean(epis),
            "epiplexity_std": np.std(epis) if len(epis) > 1 else 0,
            "final_loss_mean": np.mean(finals),
            "final_loss_std": np.std(finals) if len(finals) > 1 else 0,
            "total_steps": int(np.mean(steps)),
            # Normalized: epiplexity per step
            "epiplexity_per_step": np.mean(epis) / np.mean(steps) if np.mean(steps) > 0 else 0,
        })
    
    results.sort(key=lambda x: x["entry"])
    return results


if __name__ == "__main__":
    runs = collect_all_runs()
    print(f"Parsed {len(runs)} total runs\n")
    
    agg = aggregate_by_optimizer(runs)
    
    print(f"{'Optimizer':<25} {'Entry':>5} {'Runs':>4} {'Steps':>5} "
          f"{'Epiplexity':>12} {'±':>10} {'Final Loss':>10} {'Epi/Step':>10}")
    print("-" * 95)
    for r in agg:
        print(f"{r['optimizer']:<25} {r['entry']:>5} {r['n_runs']:>4} {r['total_steps']:>5} "
              f"{r['epiplexity_mean']:>12.1f} {r['epiplexity_std']:>10.1f} "
              f"{r['final_loss_mean']:>10.5f} {r['epiplexity_per_step']:>10.4f}")
    
    # Save raw data for plotting
    output = {
        "runs": [{k: v for k, v in r.items() if k != "points"} for r in runs],
        "aggregated": agg,
    }
    
    outdir = Path(__file__).parent
    with open(outdir / "track3_epiplexity.json", "w") as f:
        json.dump(output, f, indent=2, default=float)
    
    print(f"\nSaved to {outdir / 'track3_epiplexity.json'}")
