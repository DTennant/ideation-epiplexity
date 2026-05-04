#!/usr/bin/env python3
"""
Deep Analysis of NanoGPT Speedrun Epiplexity Data

Key improvements over TRACK1_ANALYSIS.md:
1. Fine-grained innovation classification with novelty scores
2. Sequential dependency analysis (each record builds on previous)
3. Step-normalized epiplexity to control for duration confound
4. Marginal contribution (ΔS) analysis between consecutive records
5. Cumulative innovation tracking and correlation analysis
6. Innovation diffusion / half-life analysis
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

# ─── Load data ───────────────────────────────────────────────────────────────
with open(Path(__file__).parent / 'track1_epiplexity.json') as f:
    data = json.load(f)

submissions = data['submissions']
outdir = Path(__file__).parent / 'figures'
outdir.mkdir(exist_ok=True)

# ─── Fine-grained innovation classification ──────────────────────────────────
# Each submission gets:
#   - innovation_type: fine-grained category
#   - novelty_score: 1-5 scale (1=trivial tweak, 5=paradigm shift)
#   - description: what the innovation actually does
#   - builds_on: which previous innovation(s) it directly extends

INNOVATION_DB = {
    "AdamW": {
        "type": "optimizer_core",
        "novelty": 2,  # Well-known baseline optimizer
        "desc": "Standard AdamW optimizer baseline",
        "builds_on": [],
        "paradigm": "optimizer"
    },
    "SOAP": {
        "type": "optimizer_core",
        "novelty": 4,  # Novel optimizer algorithm
        "desc": "Shampoo-family second-order optimizer",
        "builds_on": ["AdamW"],
        "paradigm": "optimizer"
    },
    "Muon": {
        "type": "optimizer_core",
        "novelty": 5,  # Genuinely novel optimizer
        "desc": "Muon optimizer — orthogonalized gradient updates via SVD, fundamentally different from Adam family",
        "builds_on": ["AdamW"],
        "paradigm": "optimizer"
    },
    "llmc": {
        "type": "training_recipe",
        "novelty": 2,  # Known technique (longer training)
        "desc": "llm.c-style long training with known schedule",
        "builds_on": ["Muon"],
        "paradigm": "training"
    },
    "ModernArch": {
        "type": "arch_backbone",
        "novelty": 3,  # Combines known components (RMSNorm, SwiGLU, etc.)
        "desc": "Modern architecture: RMSNorm, rotary embeddings, SwiGLU — known components combined",
        "builds_on": ["Muon"],
        "paradigm": "architecture"
    },
    "DistributedMuon": {
        "type": "systems_distributed",
        "novelty": 2,  # Engineering adaptation
        "desc": "Distributed version of Muon optimizer across GPUs",
        "builds_on": ["Muon", "ModernArch"],
        "paradigm": "engineering"
    },
    "PyTorch25": {
        "type": "systems_framework",
        "novelty": 1,  # Version upgrade
        "desc": "PyTorch version upgrade for better compiler support",
        "builds_on": ["DistributedMuon"],
        "paradigm": "engineering"
    },
    "ScaleUp1B": {
        "type": "arch_scale",
        "novelty": 2,  # Known scaling approach
        "desc": "Scale up to 1B parameters — larger model, more steps",
        "builds_on": ["ModernArch"],
        "paradigm": "architecture"
    },
    "Optimizers": {
        "type": "optimizer_ensemble",
        "novelty": 2,  # Trying different existing optimizers
        "desc": "Exploration of various optimizer configurations",
        "builds_on": ["Muon", "ModernArch"],
        "paradigm": "optimizer"
    },
    "UntieEmbed": {
        "type": "arch_embedding",
        "novelty": 3,  # Meaningful architectural insight
        "desc": "Untie input/output embeddings — separate representation spaces",
        "builds_on": ["ModernArch"],
        "paradigm": "architecture"
    },
    "50Bruns": {
        "type": "training_recipe",
        "novelty": 1,  # Just train longer with 50B tokens
        "desc": "Massive compute: 50B tokens, 95K steps — brute force",
        "builds_on": ["ModernArch", "Muon"],
        "paradigm": "training"
    },
    "ShortcutsTweaks": {
        "type": "arch_residual",
        "novelty": 3,  # Skip connections are known but specific design matters
        "desc": "Refined skip/shortcut connections between layers",
        "builds_on": ["ModernArch", "UntieEmbed"],
        "paradigm": "architecture"
    },
    "CastBf16": {
        "type": "systems_precision",
        "novelty": 1,  # Standard technique
        "desc": "BFloat16 casting for faster computation",
        "builds_on": ["ShortcutsTweaks"],
        "paradigm": "engineering"
    },
    "Replicateleloykun": {
        "type": "training_recipe",
        "novelty": 1,  # Replication/tuning
        "desc": "Replication of leloykun's training recipe",
        "builds_on": ["ShortcutsTweaks"],
        "paradigm": "training"
    },
    "ScaleShortcuts": {
        "type": "arch_residual",
        "novelty": 2,  # Incremental on ShortcutsTweaks
        "desc": "Learnable scaling on shortcut connections",
        "builds_on": ["ShortcutsTweaks"],
        "paradigm": "architecture"
    },
    "UNetDoubleLr": {
        "type": "arch_unet",
        "novelty": 4,  # Novel: UNet-style connections in transformers
        "desc": "UNet-style layer connections with doubled learning rate — cross-pollination from vision",
        "builds_on": ["ShortcutsTweaks"],
        "paradigm": "architecture"
    },
    "QuantizedFP4": {
        "type": "systems_quantization",
        "novelty": 3,  # FP4 quantization is moderately novel
        "desc": "FP4 quantized training for speed",
        "builds_on": ["ShortcutsTweaks", "CastBf16"],
        "paradigm": "engineering"
    },
    "FlexAttention": {
        "type": "systems_attention",
        "novelty": 2,  # Engineering optimization
        "desc": "Flexible attention implementation for better GPU utilization",
        "builds_on": ["ShortcutsTweaks"],
        "paradigm": "engineering"
    },
    "WindowWarmup": {
        "type": "arch_attention",
        "novelty": 3,  # Attention window scheduling is somewhat novel
        "desc": "Progressive attention window warmup — start local, expand to global",
        "builds_on": ["FlexAttention", "ShortcutsTweaks"],
        "paradigm": "architecture"
    },
    "ValueEmbed": {
        "type": "arch_embedding",
        "novelty": 4,  # Novel: separate value embeddings
        "desc": "Separate value embeddings for attention — new representational pathway",
        "builds_on": ["WindowWarmup", "UntieEmbed"],
        "paradigm": "architecture"
    },
    "UNetValueEmbedsTweaks": {
        "type": "arch_combined",
        "novelty": 2,  # Combining existing innovations
        "desc": "Fine-tuning UNet + ValueEmbed combination",
        "builds_on": ["ValueEmbed", "UNetDoubleLr"],
        "paradigm": "architecture"
    },
    "MFUTweaks": {
        "type": "systems_mfu",
        "novelty": 1,  # Pure engineering
        "desc": "Model FLOP Utilization optimization — same algorithm, better throughput",
        "builds_on": ["UNetValueEmbedsTweaks"],
        "paradigm": "engineering"
    },
    "SparsifyEmbeds": {
        "type": "arch_embedding",
        "novelty": 3,  # Sparsity in embeddings is interesting
        "desc": "Sparse embedding tables for efficiency",
        "builds_on": ["ValueEmbed"],
        "paradigm": "architecture"
    },
    "SoftCap": {
        "type": "arch_activation",
        "novelty": 3,  # Soft-capping logits is a specific technique
        "desc": "Soft-capped attention logits to prevent overflow — stabilization trick",
        "builds_on": ["UNetValueEmbedsTweaks"],
        "paradigm": "architecture"
    },
    "Fp8LmHead": {
        "type": "systems_precision",
        "novelty": 2,  # FP8 is somewhat novel but mostly engineering
        "desc": "FP8 precision for language model head",
        "builds_on": ["SoftCap"],
        "paradigm": "engineering"
    },
    "Sub3Min": {
        "type": "training_recipe",
        "novelty": 1,  # Milestone, not innovation
        "desc": "Sub-3-minute training — recipe tuning milestone",
        "builds_on": ["Fp8LmHead"],
        "paradigm": "training"
    },
    "BatchSize": {
        "type": "training_schedule",
        "novelty": 2,  # Known hyperparameter
        "desc": "Batch size optimization",
        "builds_on": ["Sub3Min"],
        "paradigm": "training"
    },
    "RuleTweak": {
        "type": "training_recipe",
        "novelty": 1,  # Minor rule adjustment
        "desc": "Competition rule tweaks",
        "builds_on": ["BatchSize"],
        "paradigm": "training"
    },
    "SkipMLPBlocks": {
        "type": "arch_sparsity",
        "novelty": 4,  # Selectively skipping MLP blocks is novel
        "desc": "Selective MLP block skipping — dynamic compute allocation",
        "builds_on": ["UNetValueEmbedsTweaks"],
        "paradigm": "architecture"
    },
    "FasterReduce": {
        "type": "systems_comm",
        "novelty": 1,  # AllReduce optimization
        "desc": "Faster AllReduce for gradient synchronization",
        "builds_on": ["RuleTweak"],
        "paradigm": "engineering"
    },
    "StableTorch": {
        "type": "systems_framework",
        "novelty": 1,  # Framework update
        "desc": "Stable PyTorch version for reproducibility",
        "builds_on": ["FasterReduce"],
        "paradigm": "engineering"
    },
    "EvenFasterReduce": {
        "type": "systems_comm",
        "novelty": 1,  # Incremental on FasterReduce
        "desc": "Further AllReduce optimization",
        "builds_on": ["FasterReduce"],
        "paradigm": "engineering"
    },
    "MuonWithAuxAdamExample": {
        "type": "optimizer_hybrid",
        "novelty": 3,  # Hybrid optimizer approach
        "desc": "Muon with auxiliary Adam for specific parameter groups",
        "builds_on": ["Muon"],
        "paradigm": "optimizer"
    },
    "noallreduce": {
        "type": "systems_comm",
        "novelty": 2,  # Removing allreduce is somewhat interesting
        "desc": "Eliminate AllReduce overhead via local SGD",
        "builds_on": ["EvenFasterReduce"],
        "paradigm": "engineering"
    },
    "BosAlign": {
        "type": "arch_tokenization",
        "novelty": 3,  # Novel alignment approach
        "desc": "Beginning-of-sequence alignment for better position encoding",
        "builds_on": ["SkipMLPBlocks"],
        "paradigm": "architecture"
    },
    "UpgradeTorch190": {
        "type": "systems_framework",
        "novelty": 1,
        "desc": "PyTorch 1.9.0 upgrade",
        "builds_on": ["BosAlign"],
        "paradigm": "engineering"
    },
    "TritonMuon": {
        "type": "systems_kernel",
        "novelty": 2,  # Custom Triton kernel for Muon
        "desc": "Triton kernel implementation of Muon optimizer — fused operations",
        "builds_on": ["Muon", "BosAlign"],
        "paradigm": "engineering"
    },
    "SparseAttnGate": {
        "type": "arch_attention",
        "novelty": 4,  # Gated sparse attention is novel
        "desc": "Gated sparse attention — learnable attention sparsity pattern",
        "builds_on": ["WindowWarmup", "SkipMLPBlocks"],
        "paradigm": "architecture"
    },
    "FA3": {
        "type": "systems_attention",
        "novelty": 2,  # FlashAttention 3
        "desc": "FlashAttention 3 integration for faster attention",
        "builds_on": ["SparseAttnGate"],
        "paradigm": "engineering"
    },
    "Yarn": {
        "type": "systems_kernel",
        "novelty": 1,  # Known technique
        "desc": "Yet Another RoPE extension kernel optimization",
        "builds_on": ["FA3"],
        "paradigm": "engineering"
    },
    "VectSigmoidBFloat16": {
        "type": "systems_kernel",
        "novelty": 1,
        "desc": "Vectorized sigmoid in BFloat16",
        "builds_on": ["Yarn"],
        "paradigm": "engineering"
    },
    "AsyncDataLoadAttnFinalWindow": {
        "type": "systems_pipeline",
        "novelty": 2,  # Async data loading is known but specific combo
        "desc": "Async data loading + attention final window optimization",
        "builds_on": ["VectSigmoidBFloat16"],
        "paradigm": "engineering"
    },
    "Smear": {
        "type": "arch_interpolation",
        "novelty": 3,  # Interesting interpolation technique
        "desc": "Smeared/interpolated representations between layers",
        "builds_on": ["SparseAttnGate"],
        "paradigm": "architecture"
    },
    "DropAttn": {
        "type": "arch_attention",
        "novelty": 3,  # Dropping attention heads strategically
        "desc": "Strategic attention head dropout during training",
        "builds_on": ["SparseAttnGate", "Smear"],
        "paradigm": "architecture"
    },
    "MuonCustomSizing": {
        "type": "optimizer_tuning",
        "novelty": 2,  # Hyperparameter tuning of Muon
        "desc": "Custom parameter group sizing for Muon optimizer",
        "builds_on": ["Muon", "TritonMuon"],
        "paradigm": "optimizer"
    },
    "BF16CE": {
        "type": "systems_precision",
        "novelty": 1,
        "desc": "BFloat16 cross-entropy computation",
        "builds_on": ["MuonCustomSizing"],
        "paradigm": "engineering"
    },
    "PolarExpress": {
        "type": "optimizer_core",
        "novelty": 3,  # Novel optimizer variant
        "desc": "Polar decomposition-based optimizer updates",
        "builds_on": ["Muon"],
        "paradigm": "optimizer"
    },
    "CustomBatching": {
        "type": "training_schedule",
        "novelty": 2,
        "desc": "Custom batching strategy for variable-length sequences",
        "builds_on": ["PolarExpress"],
        "paradigm": "engineering"
    },
    "Backout": {
        "type": "arch_residual",
        "novelty": 3,  # Interesting residual modification
        "desc": "Backout mechanism — selective residual connection disabling",
        "builds_on": ["ShortcutsTweaks", "CustomBatching"],
        "paradigm": "architecture"
    },
    "NorMuon": {
        "type": "optimizer_core",
        "novelty": 4,  # Significant Muon variant
        "desc": "Normalized Muon — gradient normalization before orthogonalization",
        "builds_on": ["Muon"],
        "paradigm": "optimizer"
    },
    "FixMuonLR": {
        "type": "optimizer_tuning",
        "novelty": 1,  # Bug fix / tuning
        "desc": "Fix Muon learning rate schedule bug",
        "builds_on": ["NorMuon"],
        "paradigm": "optimizer"
    },
    "AdamSyncGradientHook": {
        "type": "optimizer_engineering",
        "novelty": 2,  # Engineering in optimizer
        "desc": "Synchronized gradient hooks for Adam optimizer",
        "builds_on": ["NorMuon"],
        "paradigm": "optimizer"
    },
    "CautiousWD": {
        "type": "optimizer_regularization",
        "novelty": 3,  # Cautious weight decay is a novel technique
        "desc": "Cautious weight decay — only apply WD when gradient agrees with weight direction",
        "builds_on": ["NorMuon"],
        "paradigm": "optimizer"
    },
    "RefineSkip": {
        "type": "arch_residual",
        "novelty": 2,  # Refinement of existing skip connections
        "desc": "Refined skip connection patterns",
        "builds_on": ["ShortcutsTweaks", "Backout"],
        "paradigm": "architecture"
    },
    "BatchSizeSchedule": {
        "type": "training_schedule",
        "novelty": 3,  # Dynamic batch sizing is interesting
        "desc": "Dynamic batch size scheduling during training",
        "builds_on": ["BatchSize", "NorMuon"],
        "paradigm": "training"
    },
    "SALambdaOnWeights": {
        "type": "arch_regularization",
        "novelty": 3,
        "desc": "Spectral regularization via lambda on weights — controls weight spectrum",
        "builds_on": ["NorMuon", "RefineSkip"],
        "paradigm": "architecture"
    },
    "NorMuonOptimsAndFixes": {
        "type": "optimizer_tuning",
        "novelty": 2,
        "desc": "NorMuon optimizations and bug fixes",
        "builds_on": ["NorMuon"],
        "paradigm": "optimizer"
    },
    "PartialKeyOffset": {
        "type": "arch_attention",
        "novelty": 4,  # Novel attention modification
        "desc": "Partial key offset — offset a subset of attention keys for better diversity",
        "builds_on": ["SparseAttnGate", "WindowWarmup"],
        "paradigm": "architecture"
    },
    "CautiousWDAdam": {
        "type": "optimizer_regularization",
        "novelty": 2,  # Applying CautiousWD to Adam
        "desc": "Cautious weight decay applied to Adam parameter groups",
        "builds_on": ["CautiousWD"],
        "paradigm": "optimizer"
    },
    "RetieLMHead": {
        "type": "arch_embedding",
        "novelty": 2,  # Reverse of UntieEmbed
        "desc": "Re-tie language model head weights — reversal based on new context",
        "builds_on": ["UntieEmbed"],
        "paradigm": "architecture"
    },
    "SmoothedScalars": {
        "type": "optimizer_tuning",
        "novelty": 2,
        "desc": "Smoothed scalar hyperparameters during training",
        "builds_on": ["NorMuonOptimsAndFixes"],
        "paradigm": "optimizer"
    },
    "MultiTokenPrediction": {
        "type": "arch_objective",
        "novelty": 5,  # Novel training objective
        "desc": "Multi-token prediction — predict multiple future tokens simultaneously, fundamentally different objective",
        "builds_on": ["UNetValueEmbedsTweaks"],
        "paradigm": "architecture"
    },
    "LogitRescale": {
        "type": "arch_activation",
        "novelty": 2,  # Known technique
        "desc": "Rescale output logits for training stability",
        "builds_on": ["MultiTokenPrediction"],
        "paradigm": "architecture"
    },
    "VeSkipGates": {
        "type": "arch_gating",
        "novelty": 3,  # Gating mechanism
        "desc": "Value embedding skip gates — learned gating for residual value streams",
        "builds_on": ["ValueEmbed", "LogitRescale"],
        "paradigm": "architecture"
    },
    "GatesToCompiledAdam": {
        "type": "optimizer_engineering",
        "novelty": 1,  # Engineering
        "desc": "Compile Adam optimizer for gates — pure speed optimization",
        "builds_on": ["VeSkipGates"],
        "paradigm": "engineering"
    },
    "MixedPrecisionInterweavedOptimizer": {
        "type": "optimizer_hybrid",
        "novelty": 3,  # Interesting hybrid approach
        "desc": "Interleaved mixed-precision optimizer — different precisions for different parameter groups",
        "builds_on": ["NorMuon", "GatesToCompiledAdam"],
        "paradigm": "optimizer"
    },
    "PairedHeadAttention": {
        "type": "arch_attention",
        "novelty": 4,  # Novel attention variant
        "desc": "Paired-head attention — pairs of heads share keys, halving key computation",
        "builds_on": ["SparseAttnGate", "PartialKeyOffset"],
        "paradigm": "architecture"
    },
    "FusedLinearReLUSquare": {
        "type": "systems_kernel",
        "novelty": 1,  # Fusing known ops
        "desc": "Fused Linear + ReLU² kernel — ReLU² itself is known, fusing is engineering",
        "builds_on": ["PairedHeadAttention"],
        "paradigm": "engineering"
    },
    "FusedSoftcappedEntropy": {
        "type": "systems_kernel",
        "novelty": 1,
        "desc": "Fused soft-capped cross-entropy kernel",
        "builds_on": ["SoftCap", "FusedLinearReLUSquare"],
        "paradigm": "engineering"
    },
    "UnifiedOptimizers": {
        "type": "optimizer_engineering",
        "novelty": 2,
        "desc": "Unified optimizer interface for Muon + Adam parameter groups",
        "builds_on": ["NorMuon", "MixedPrecisionInterweavedOptimizer"],
        "paradigm": "optimizer"
    },
    "BigramHashEmbedding": {
        "type": "arch_embedding",
        "novelty": 4,  # Novel embedding approach
        "desc": "Bigram hash embedding — hash-based subword embeddings capturing bigram statistics",
        "builds_on": ["ValueEmbed", "SparsifyEmbeds"],
        "paradigm": "architecture"
    },
    "ImprovedLMHead": {
        "type": "arch_head",
        "novelty": 3,
        "desc": "Improved language model head architecture",
        "builds_on": ["RetieLMHead", "BigramHashEmbedding"],
        "paradigm": "architecture"
    },
    "UntieValueEmbeddings": {
        "type": "arch_embedding",
        "novelty": 3,
        "desc": "Untied value embeddings — independent value projection weights",
        "builds_on": ["ValueEmbed", "UntieEmbed"],
        "paradigm": "architecture"
    },
    "MimeticValueOutput": {
        "type": "arch_initialization",
        "novelty": 4,  # Mimetic initialization is novel
        "desc": "Mimetic initialization for value/output projections — initialize to mimic target function",
        "builds_on": ["UntieValueEmbeddings"],
        "paradigm": "architecture"
    },
    "VeFused": {
        "type": "systems_kernel",
        "novelty": 1,
        "desc": "Fused value embedding kernel",
        "builds_on": ["ValueEmbed", "FusedLinearReLUSquare"],
        "paradigm": "engineering"
    },
    "BigramHashH2D": {
        "type": "systems_kernel",
        "novelty": 1,
        "desc": "Host-to-device optimization for bigram hash embeddings",
        "builds_on": ["BigramHashEmbedding"],
        "paradigm": "engineering"
    },
    "KernelTuning": {
        "type": "systems_kernel",
        "novelty": 1,
        "desc": "General kernel auto-tuning",
        "builds_on": ["BigramHashH2D"],
        "paradigm": "engineering"
    },
    "VeTuned": {
        "type": "systems_kernel",
        "novelty": 1,
        "desc": "Value embedding kernel tuning",
        "builds_on": ["VeFused"],
        "paradigm": "engineering"
    },
    "SparseBigramGradient": {
        "type": "systems_kernel",
        "novelty": 2,
        "desc": "Sparse gradient computation for bigram embeddings",
        "builds_on": ["BigramHashEmbedding"],
        "paradigm": "engineering"
    },
    "ShortWindow": {
        "type": "arch_attention",
        "novelty": 2,
        "desc": "Shorter attention window for efficiency",
        "builds_on": ["WindowWarmup"],
        "paradigm": "engineering"
    },
    "ParallelResiduals": {
        "type": "arch_residual",
        "novelty": 3,  # Parallel residual streams
        "desc": "Parallel residual streams — run attention and MLP in parallel instead of sequential",
        "builds_on": ["ShortcutsTweaks", "RefineSkip"],
        "paradigm": "architecture"
    },
    "FlattenForward": {
        "type": "systems_kernel",
        "novelty": 1,
        "desc": "Flattened forward pass for better GPU utilization",
        "builds_on": ["ParallelResiduals"],
        "paradigm": "engineering"
    },
    "CrossEntropyKernel": {
        "type": "systems_kernel",
        "novelty": 1,
        "desc": "Custom cross-entropy kernel",
        "builds_on": ["FlattenForward"],
        "paradigm": "engineering"
    },
    "TransposeCopyBackward": {
        "type": "systems_kernel",
        "novelty": 1,
        "desc": "Optimized transpose+copy in backward pass",
        "builds_on": ["CrossEntropyKernel"],
        "paradigm": "engineering"
    },
    "SimplifyHC": {
        "type": "systems_refactor",
        "novelty": 1,
        "desc": "Simplify hyperparameter configuration code",
        "builds_on": ["TransposeCopyBackward"],
        "paradigm": "engineering"
    },
    "VarlenMaxDocs": {
        "type": "systems_data",
        "novelty": 2,
        "desc": "Variable-length sequences with max documents packing",
        "builds_on": ["SimplifyHC"],
        "paradigm": "engineering"
    },
    "FuseCEFwdAndBwd": {
        "type": "systems_kernel",
        "novelty": 1,
        "desc": "Fuse cross-entropy forward and backward passes",
        "builds_on": ["CrossEntropyKernel"],
        "paradigm": "engineering"
    },
    "PairedHeadMuon": {
        "type": "optimizer_arch_integration",
        "novelty": 2,
        "desc": "Muon optimizer integration tuned for paired-head attention",
        "builds_on": ["PairedHeadAttention", "NorMuon"],
        "paradigm": "engineering"
    },
}

# ─── Annotate submissions ────────────────────────────────────────────────────
for sub in submissions:
    label = sub['label']
    if label in INNOVATION_DB:
        info = INNOVATION_DB[label]
        sub['innovation_type'] = info['type']
        sub['novelty'] = info['novelty']
        sub['description'] = info['desc']
        sub['paradigm'] = info['paradigm']
        sub['builds_on'] = info['builds_on']
    else:
        sub['innovation_type'] = 'unknown'
        sub['novelty'] = 1
        sub['description'] = ''
        sub['paradigm'] = 'unknown'
        sub['builds_on'] = []

# ─── Compute derived metrics ─────────────────────────────────────────────────
for sub in submissions:
    sub['epiplexity_per_step'] = sub['epiplexity_mean'] / sub['total_steps'] if sub['total_steps'] > 0 else 0

# Sequential delta
for i, sub in enumerate(submissions):
    if i == 0:
        sub['delta_epiplexity'] = 0
        sub['delta_normalized'] = 0
        sub['delta_steps'] = 0
    else:
        prev = submissions[i-1]
        sub['delta_epiplexity'] = sub['epiplexity_mean'] - prev['epiplexity_mean']
        sub['delta_normalized'] = sub['epiplexity_per_step'] - prev['epiplexity_per_step']
        sub['delta_steps'] = sub['total_steps'] - prev['total_steps']

# Cumulative novelty (sum of all novelty scores up to this point)
cumulative = 0
for sub in submissions:
    cumulative += sub['novelty']
    sub['cumulative_novelty'] = cumulative

# Cumulative high-novelty count (novelty >= 4)
cum_high = 0
for sub in submissions:
    if sub['novelty'] >= 4:
        cum_high += 1
    sub['cumulative_high_novelty'] = cum_high

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Fine-grained innovation type vs normalized epiplexity
# ═══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 7))

# Group by innovation type
type_data = defaultdict(list)
for sub in submissions:
    type_data[sub['innovation_type']].append(sub['epiplexity_per_step'])

# Sort by median
sorted_types = sorted(type_data.keys(), key=lambda t: np.median(type_data[t]), reverse=True)

# Color by paradigm
paradigm_colors = {
    'optimizer': '#e74c3c',
    'architecture': '#3498db',
    'engineering': '#2ecc71',
    'training': '#f39c12',
    'unknown': '#95a5a6'
}

positions = range(len(sorted_types))
for i, t in enumerate(sorted_types):
    vals = type_data[t]
    # Determine paradigm from first submission with this type
    paradigm = 'unknown'
    for sub in submissions:
        if sub['innovation_type'] == t:
            paradigm = sub['paradigm']
            break
    color = paradigm_colors.get(paradigm, '#95a5a6')
    
    bp = ax.boxplot([vals], positions=[i], widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor=color, alpha=0.6),
                    medianprops=dict(color='black', linewidth=2))

ax.set_xticks(range(len(sorted_types)))
ax.set_xticklabels([t.replace('_', '\n') for t in sorted_types], rotation=45, ha='right', fontsize=7)
ax.set_ylabel('Epiplexity per Step (Normalized)', fontsize=12)
ax.set_title('Normalized Epiplexity by Fine-Grained Innovation Type', fontsize=14)

# Legend
legend_patches = [mpatches.Patch(color=c, alpha=0.6, label=p.title()) 
                  for p, c in paradigm_colors.items() if p != 'unknown']
ax.legend(handles=legend_patches, loc='upper right')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(outdir / 'fig1_innovation_type_normalized.png', dpi=150, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Novelty score vs normalized epiplexity scatter
# ═══════════════════════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: scatter with jitter
novelties = [sub['novelty'] for sub in submissions]
eps_norm = [sub['epiplexity_per_step'] for sub in submissions]
categories_orig = [sub['category'] for sub in submissions]

cat_colors = {'Architecture': '#3498db', 'Optimization': '#e74c3c', 'Engineering': '#2ecc71'}
colors = [cat_colors[c] for c in categories_orig]

# Add jitter to novelty for visibility
jitter = np.random.normal(0, 0.08, len(novelties))
ax1.scatter([n + j for n, j in zip(novelties, jitter)], eps_norm, 
            c=colors, alpha=0.6, s=50, edgecolors='black', linewidth=0.3)

# Add median line per novelty level
for n in range(1, 6):
    vals = [sub['epiplexity_per_step'] for sub in submissions if sub['novelty'] == n]
    if vals:
        ax1.plot([n-0.3, n+0.3], [np.median(vals)]*2, 'k-', linewidth=2.5)
        ax1.text(n+0.35, np.median(vals), f'{np.median(vals):.3f}', fontsize=8, va='center')

ax1.set_xlabel('Novelty Score', fontsize=12)
ax1.set_ylabel('Epiplexity per Step', fontsize=12)
ax1.set_title('Novelty Score vs Normalized Epiplexity', fontsize=13)
ax1.set_xticks(range(1, 6))
ax1.set_xticklabels(['1\nTrivial', '2\nIncremental', '3\nModerate', '4\nNovel', '5\nParadigm'])
ax1.grid(alpha=0.3)

legend_patches = [mpatches.Patch(color=c, label=cat) for cat, c in cat_colors.items()]
ax1.legend(handles=legend_patches, fontsize=9)

# Right: novelty histogram with example labels
novelty_counts = defaultdict(int)
novelty_examples = defaultdict(list)
for sub in submissions:
    novelty_counts[sub['novelty']] += 1
    if sub['novelty'] >= 3:
        novelty_examples[sub['novelty']].append(sub['label'])

bars = ax2.bar(range(1, 6), [novelty_counts.get(i, 0) for i in range(1, 6)],
               color=['#d5d5d5', '#b0b0b0', '#ffd700', '#ff8c00', '#ff4500'],
               edgecolor='black', linewidth=0.5)

ax2.set_xlabel('Novelty Score', fontsize=12)
ax2.set_ylabel('Number of Submissions', fontsize=12)
ax2.set_title('Distribution of Innovation Novelty', fontsize=13)
ax2.set_xticks(range(1, 6))
ax2.set_xticklabels(['1\nTrivial', '2\nIncremental', '3\nModerate', '4\nNovel', '5\nParadigm'])

# Annotate with examples
for n in [4, 5]:
    examples = novelty_examples.get(n, [])[:4]
    if examples:
        txt = '\n'.join(examples)
        ax2.annotate(txt, xy=(n, novelty_counts[n]), 
                    xytext=(n + 0.5, novelty_counts[n] + 2),
                    fontsize=7, ha='left',
                    arrowprops=dict(arrowstyle='->', color='gray'),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(outdir / 'fig2_novelty_vs_epiplexity.png', dpi=150, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Sequential dependency — marginal ΔS and cumulative novelty
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)

indices = range(len(submissions))
labels = [sub['label'] for sub in submissions]

# Panel A: Normalized epiplexity over sequence
eps_per_step = [sub['epiplexity_per_step'] for sub in submissions]
novelty_scores = [sub['novelty'] for sub in submissions]
paradigms = [sub['paradigm'] for sub in submissions]
p_colors = [paradigm_colors.get(p, '#95a5a6') for p in paradigms]

axes[0].bar(indices, eps_per_step, color=p_colors, alpha=0.7, edgecolor='black', linewidth=0.3)

# Highlight high-novelty submissions
for i, sub in enumerate(submissions):
    if sub['novelty'] >= 4:
        axes[0].annotate(sub['label'], xy=(i, eps_per_step[i]),
                        xytext=(i, eps_per_step[i] + 0.03),
                        fontsize=6, rotation=45, ha='left', va='bottom',
                        color='red', fontweight='bold')

axes[0].set_ylabel('Epiplexity / Step', fontsize=11)
axes[0].set_title('A. Step-Normalized Epiplexity Across Speedrun History', fontsize=13)
axes[0].grid(axis='y', alpha=0.3)

# Panel B: Marginal ΔS (normalized) between consecutive submissions
delta_norm = [sub['delta_normalized'] for sub in submissions]
delta_colors = ['#e74c3c' if d > 0 else '#2ecc71' for d in delta_norm]
axes[1].bar(indices, delta_norm, color=delta_colors, alpha=0.7, edgecolor='black', linewidth=0.3)

# Mark innovations with novelty >= 4
for i, sub in enumerate(submissions):
    if sub['novelty'] >= 4:
        axes[1].axvline(x=i, color='gold', linestyle='--', alpha=0.7, linewidth=1)
        axes[1].annotate(sub['label'], xy=(i, delta_norm[i]),
                        xytext=(i, max(delta_norm) * 0.8),
                        fontsize=6, rotation=90, ha='center', va='bottom',
                        color='darkorange', fontweight='bold')

axes[1].set_ylabel('Δ(Epiplexity/Step)', fontsize=11)
axes[1].set_title('B. Marginal Change in Normalized Epiplexity (Red=Increase, Green=Decrease)', fontsize=13)
axes[1].axhline(y=0, color='black', linewidth=0.5)
axes[1].grid(axis='y', alpha=0.3)

# Panel C: Cumulative novelty score
cum_novelty = [sub['cumulative_novelty'] for sub in submissions]
cum_high = [sub['cumulative_high_novelty'] for sub in submissions]

ax3_twin = axes[2].twinx()
axes[2].plot(indices, cum_novelty, 'b-o', markersize=3, linewidth=1.5, label='Cumulative Novelty (all)')
ax3_twin.plot(indices, cum_high, 'r-s', markersize=3, linewidth=1.5, label='Cumulative High-Novelty (≥4)')
ax3_twin.fill_between(indices, cum_high, alpha=0.1, color='red')

axes[2].set_xlabel('Submission Index', fontsize=11)
axes[2].set_ylabel('Cumulative Novelty Score', fontsize=11, color='blue')
ax3_twin.set_ylabel('Cumulative High-Novelty Count', fontsize=11, color='red')
axes[2].set_title('C. Cumulative Innovation Across Speedrun History', fontsize=13)

# Add x-axis labels for every 5th submission
tick_indices = list(range(0, len(submissions), 5))
axes[2].set_xticks(tick_indices)
axes[2].set_xticklabels([f'{i}\n{labels[i]}' for i in tick_indices], rotation=45, ha='right', fontsize=7)

lines1, labels1 = axes[2].get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
axes[2].legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig(outdir / 'fig3_sequential_dependency.png', dpi=150, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Innovation diffusion — how do high-novelty ideas affect subsequent records?
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Identify high-novelty submissions (novelty >= 4)
high_novelty_indices = [i for i, sub in enumerate(submissions) if sub['novelty'] >= 4]
high_novelty_labels = [submissions[i]['label'] for i in high_novelty_indices]

# For each high-novelty innovation, track normalized epiplexity of the next N submissions
N_AFTER = 10
colors_hn = plt.cm.Set1(np.linspace(0, 1, len(high_novelty_indices)))

for idx, (hn_i, color) in enumerate(zip(high_novelty_indices, colors_hn)):
    hn_label = submissions[hn_i]['label']
    # Get next N submissions
    after_indices = list(range(hn_i, min(hn_i + N_AFTER + 1, len(submissions))))
    after_eps = [submissions[i]['epiplexity_per_step'] for i in after_indices]
    # Normalize relative to the innovation's own epiplexity/step
    baseline = after_eps[0]
    relative = [(e - baseline) for e in after_eps]
    
    axes[0].plot(range(len(relative)), relative, '-o', color=color, 
                label=hn_label, markersize=4, linewidth=1.5, alpha=0.8)

axes[0].axhline(y=0, color='black', linewidth=0.5, linestyle='--')
axes[0].set_xlabel('Submissions After Innovation', fontsize=11)
axes[0].set_ylabel('Δ(Epiplexity/Step) Relative to Innovation', fontsize=11)
axes[0].set_title('A. Innovation Diffusion: How Normalized Epiplexity Changes After High-Novelty Innovations', fontsize=12)
axes[0].legend(fontsize=8, ncol=2, loc='upper right')
axes[0].grid(alpha=0.3)

# Panel B: Correlation — cumulative high novelty vs normalized epiplexity
ax2 = axes[1]
cum_high_arr = np.array([sub['cumulative_high_novelty'] for sub in submissions])
eps_norm_arr = np.array([sub['epiplexity_per_step'] for sub in submissions])

# Exclude extreme outliers (50Bruns, llmc, ScaleUp1B)
mask = np.array([sub['label'] not in ['50Bruns', 'llmc', 'ScaleUp1B'] for sub in submissions])
x_clean = cum_high_arr[mask]
y_clean = eps_norm_arr[mask]

ax2.scatter(x_clean, y_clean, c=[paradigm_colors.get(submissions[i]['paradigm'], '#95a5a6') 
            for i in range(len(submissions)) if mask[i]], 
            alpha=0.6, s=50, edgecolors='black', linewidth=0.3)

# Fit regression line
z = np.polyfit(x_clean, y_clean, 1)
p = np.poly1d(z)
x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
ax2.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7)

# Correlation
corr = np.corrcoef(x_clean, y_clean)[0, 1]
ax2.text(0.05, 0.95, f'Pearson r = {corr:.3f}', transform=ax2.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax2.set_xlabel('Cumulative High-Novelty Innovations (≥4)', fontsize=11)
ax2.set_ylabel('Epiplexity per Step', fontsize=11)
ax2.set_title('B. Cumulative High-Novelty Innovations vs Normalized Epiplexity', fontsize=12)
ax2.grid(alpha=0.3)

legend_patches = [mpatches.Patch(color=c, alpha=0.6, label=p.title()) 
                  for p, c in paradigm_colors.items() if p != 'unknown']
ax2.legend(handles=legend_patches, fontsize=9)

plt.tight_layout()
plt.savefig(outdir / 'fig4_innovation_diffusion.png', dpi=150, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Phase analysis — distinct phases of the speedrun
# ═══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(18, 8))

# Define phases based on step count regimes and innovation patterns
phases = [
    (0, 4, "Phase 1:\nExploration\n(Jun-Oct 2024)", '#ffcccc'),
    (4, 11, "Phase 2:\nArchitecture\nRevolution\n(Oct-Nov 2024)", '#cce5ff'),
    (11, 20, "Phase 3:\nRapid\nConvergence\n(Nov-Dec 2024)", '#ccffcc'),
    (20, 28, "Phase 4:\nPlateau &\nPolish\n(Dec 2024-Feb 2025)", '#ffffcc'),
    (28, 48, "Phase 5:\nMicro-\nOptimization\n(May-Sep 2025)", '#e5ccff'),
    (48, 67, "Phase 6:\nSecond Wind\n(Sep 2025-Jan 2026)", '#ffd9cc'),
    (67, 89, "Phase 7:\nFinal Push\n(Jan-Apr 2026)", '#ccffe5'),
]

# Plot normalized epiplexity
x = range(len(submissions))
y = [sub['epiplexity_per_step'] for sub in submissions]

# Draw phase backgrounds
for start, end, label, color in phases:
    end = min(end, len(submissions))
    ax.axvspan(start - 0.5, end - 0.5, alpha=0.2, color=color)
    mid = (start + end) / 2
    ax.text(mid, max(y) * 0.95, label, ha='center', va='top', fontsize=7,
            fontweight='bold', style='italic')

# Plot bars colored by novelty
novelty_cmap = {1: '#d5d5d5', 2: '#b0b0b0', 3: '#ffd700', 4: '#ff8c00', 5: '#ff4500'}
bar_colors = [novelty_cmap[sub['novelty']] for sub in submissions]
ax.bar(x, y, color=bar_colors, edgecolor='black', linewidth=0.3, alpha=0.8)

# Annotate key innovations
key_innovations = {
    'Muon': (5, 'Muon optimizer'),
    'ValueEmbed': (4, 'Value embeddings'),
    'MultiTokenPrediction': (5, 'Multi-token prediction'),
    'NorMuon': (4, 'NorMuon'),
    'PairedHeadAttention': (4, 'Paired-head attn'),
    'SkipMLPBlocks': (4, 'Skip MLP blocks'),
    'SparseAttnGate': (4, 'Sparse attn gate'),
    'BigramHashEmbedding': (4, 'Bigram hash embed'),
}

for i, sub in enumerate(submissions):
    if sub['label'] in key_innovations:
        nov, short_label = key_innovations[sub['label']]
        ax.annotate(short_label, xy=(i, y[i]),
                   xytext=(i, y[i] + 0.08),
                   fontsize=6, rotation=60, ha='left', va='bottom',
                   color='darkred', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='darkred', lw=0.8))

ax.set_xlabel('Submission Index', fontsize=12)
ax.set_ylabel('Epiplexity per Step', fontsize=12)
ax.set_title('NanoGPT Speedrun Phases: Normalized Epiplexity Colored by Innovation Novelty', fontsize=14)

# Novelty legend
nov_patches = [mpatches.Patch(color=novelty_cmap[n], label=f'Novelty {n}') for n in range(1, 6)]
ax.legend(handles=nov_patches, loc='upper right', fontsize=8)
ax.grid(axis='y', alpha=0.3)

# X-axis with selected labels
tick_idx = list(range(0, len(submissions), 5))
ax.set_xticks(tick_idx)
ax.set_xticklabels([f'{i}: {submissions[i]["label"]}' for i in tick_idx], rotation=45, ha='right', fontsize=7)

plt.tight_layout()
plt.savefig(outdir / 'fig5_phase_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Innovation dependency graph (simplified) + temporal correlation
# ═══════════════════════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Panel A: Heatmap of normalized epiplexity in 2D (time × innovation type)
# Group by paradigm over time
paradigm_timeline = defaultdict(list)
for sub in submissions:
    paradigm_timeline[sub['paradigm']].append(sub['epiplexity_per_step'])

paradigm_order = ['optimizer', 'architecture', 'engineering', 'training']
# Create a window-averaged version
window = 5
paradigm_medians = {}
for p in paradigm_order:
    vals = []
    for i in range(0, len(submissions), window):
        chunk = [sub['epiplexity_per_step'] for sub in submissions[i:i+window] if sub['paradigm'] == p]
        vals.append(np.mean(chunk) if chunk else np.nan)
    paradigm_medians[p] = vals

n_windows = len(paradigm_medians['optimizer'])
data_matrix = np.array([paradigm_medians[p] for p in paradigm_order])

im = ax1.imshow(data_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax1.set_yticks(range(len(paradigm_order)))
ax1.set_yticklabels([p.title() for p in paradigm_order])
ax1.set_xlabel(f'Time Window (every {window} submissions)', fontsize=11)
ax1.set_title('A. Paradigm × Time: Mean Epiplexity/Step', fontsize=12)
plt.colorbar(im, ax=ax1, shrink=0.8, label='Epiplexity / Step')

# Panel B: Auto-correlation of normalized epiplexity series
eps_series = np.array([sub['epiplexity_per_step'] for sub in submissions])
# Remove outliers for cleaner autocorrelation
mask_clean = np.array([sub['label'] not in ['50Bruns', 'llmc', 'ScaleUp1B'] for sub in submissions])
eps_clean = eps_series[mask_clean]

max_lag = 20
autocorr = []
for lag in range(max_lag + 1):
    if lag == 0:
        autocorr.append(1.0)
    else:
        c = np.corrcoef(eps_clean[:-lag], eps_clean[lag:])[0, 1]
        autocorr.append(c)

ax2.bar(range(max_lag + 1), autocorr, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.3)
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.axhline(y=1.96/np.sqrt(len(eps_clean)), color='red', linewidth=1, linestyle='--', alpha=0.7, label='95% CI')
ax2.axhline(y=-1.96/np.sqrt(len(eps_clean)), color='red', linewidth=1, linestyle='--', alpha=0.7)

ax2.set_xlabel('Lag (submissions)', fontsize=11)
ax2.set_ylabel('Autocorrelation', fontsize=11)
ax2.set_title('B. Autocorrelation of Normalized Epiplexity\n(Sequential Dependency Evidence)', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(outdir / 'fig6_paradigm_heatmap_autocorr.png', dpi=150, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: "Innovation Genealogy" — which innovations build on which?
# ═══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(16, 10))

# Track "active innovations" — all innovations that are present in the codebase at each point
# (each record builds on previous, so innovations accumulate)
# Categorize accumulated innovations by paradigm

accumulated_by_paradigm = {'optimizer': [], 'architecture': [], 'engineering': [], 'training': []}
accumulated_novelty_by_paradigm = {'optimizer': [], 'architecture': [], 'engineering': [], 'training': []}

optimizer_innovations = set()
arch_innovations = set()
eng_innovations = set()
train_innovations = set()

paradigm_sets = {
    'optimizer': optimizer_innovations,
    'architecture': arch_innovations,
    'engineering': eng_innovations,
    'training': train_innovations,
}

for sub in submissions:
    p = sub['paradigm']
    if p in paradigm_sets:
        paradigm_sets[p].add(sub['label'])
    
    for pp in ['optimizer', 'architecture', 'engineering', 'training']:
        accumulated_by_paradigm[pp].append(len(paradigm_sets[pp]))

    # Also track weighted by novelty
    for pp in ['optimizer', 'architecture', 'engineering', 'training']:
        total_novelty = sum(
            INNOVATION_DB.get(lbl, {}).get('novelty', 1) 
            for lbl in paradigm_sets[pp]
        )
        accumulated_novelty_by_paradigm[pp].append(total_novelty)

x_vals = range(len(submissions))

# Stacked area chart of accumulated innovations
bottom = np.zeros(len(submissions))
paradigm_display_order = ['engineering', 'training', 'optimizer', 'architecture']
paradigm_display_colors = {
    'engineering': '#2ecc71', 'training': '#f39c12', 
    'optimizer': '#e74c3c', 'architecture': '#3498db'
}

for pp in paradigm_display_order:
    vals = np.array(accumulated_novelty_by_paradigm[pp])
    ax.fill_between(x_vals, bottom, bottom + vals, alpha=0.5, 
                    color=paradigm_display_colors[pp], label=f'{pp.title()} (novelty-weighted)')
    bottom += vals

ax.set_xlabel('Submission Index', fontsize=12)
ax.set_ylabel('Accumulated Innovation Score (Novelty-Weighted)', fontsize=12)
ax.set_title('Innovation Genealogy: Cumulative Novelty-Weighted Innovations by Paradigm', fontsize=14)
ax.legend(loc='upper left', fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Annotate milestone innovations
milestones = ['Muon', 'ModernArch', 'ValueEmbed', 'NorMuon', 'MultiTokenPrediction', 'PairedHeadAttention']
for i, sub in enumerate(submissions):
    if sub['label'] in milestones:
        total_at_point = sum(accumulated_novelty_by_paradigm[pp][i] for pp in paradigm_display_order)
        ax.annotate(sub['label'], xy=(i, total_at_point),
                   xytext=(i + 2, total_at_point + 5),
                   fontsize=7, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

tick_idx = list(range(0, len(submissions), 10))
ax.set_xticks(tick_idx)
ax.set_xticklabels([f'{i}: {submissions[i]["label"]}' for i in tick_idx], rotation=45, ha='right', fontsize=8)

plt.tight_layout()
plt.savefig(outdir / 'fig7_innovation_genealogy.png', dpi=150, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 8: The key insight — normalized epiplexity INCREASES in late phases
# ═══════════════════════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# Panel A: Raw epiplexity vs step count (showing the confound)
steps = [sub['total_steps'] for sub in submissions]
eps_raw = [sub['epiplexity_mean'] for sub in submissions]

# Exclude extreme outliers
mask = [sub['label'] not in ['50Bruns', 'llmc', 'ScaleUp1B'] for sub in submissions]
steps_clean = [s for s, m in zip(steps, mask) if m]
eps_clean = [e for e, m in zip(eps_raw, mask) if m]
labels_clean = [sub['label'] for sub, m in zip(submissions, mask) if m]
novelty_clean = [sub['novelty'] for sub, m in zip(submissions, mask) if m]

sc = ax1.scatter(steps_clean, eps_clean, c=novelty_clean, cmap='YlOrRd', 
                s=60, edgecolors='black', linewidth=0.3, vmin=1, vmax=5)
plt.colorbar(sc, ax=ax1, label='Novelty Score', shrink=0.8)

# Fit line
z = np.polyfit(steps_clean, eps_clean, 1)
p = np.poly1d(z)
x_fit = np.linspace(min(steps_clean), max(steps_clean), 100)
ax1.plot(x_fit, p(x_fit), 'r--', linewidth=2, alpha=0.7)
corr = np.corrcoef(steps_clean, eps_clean)[0, 1]
ax1.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax1.transAxes, fontsize=12,
        va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax1.set_xlabel('Total Steps', fontsize=11)
ax1.set_ylabel('Raw Epiplexity', fontsize=11)
ax1.set_title('A. The Confound: Raw Epiplexity is Dominated by Step Count', fontsize=13)
ax1.grid(alpha=0.3)

# Panel B: Timeline of epiplexity per step — reveals the interesting late-phase increase
eps_per_step_clean = [sub['epiplexity_per_step'] for sub, m in zip(submissions, mask) if m]
indices_clean = range(len(eps_per_step_clean))
novelty_clean_arr = [sub['novelty'] for sub, m in zip(submissions, mask) if m]

ax2.plot(indices_clean, eps_per_step_clean, 'b-', alpha=0.3, linewidth=1)
sc2 = ax2.scatter(indices_clean, eps_per_step_clean, c=novelty_clean_arr, 
                  cmap='YlOrRd', s=40, edgecolors='black', linewidth=0.3, vmin=1, vmax=5, zorder=5)

# Rolling average
window = 5
rolling = np.convolve(eps_per_step_clean, np.ones(window)/window, mode='valid')
ax2.plot(range(window//2, window//2 + len(rolling)), rolling, 'k-', linewidth=2.5, label=f'{window}-pt moving avg')

# Mark the dip and rise
min_idx = np.argmin(eps_per_step_clean)
ax2.annotate(f'Minimum: {eps_per_step_clean[min_idx]:.3f}\n({labels_clean[min_idx]})',
            xy=(min_idx, eps_per_step_clean[min_idx]),
            xytext=(min_idx + 5, eps_per_step_clean[min_idx] - 0.05),
            fontsize=9, arrowprops=dict(arrowstyle='->', color='blue'),
            color='blue')

ax2.set_xlabel('Submission Index (outliers excluded)', fontsize=11)
ax2.set_ylabel('Epiplexity per Step', fontsize=11)
ax2.set_title('B. The Surprise: Normalized Epiplexity Increases in Late Phases (More Learning Per Step)', fontsize=13)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(outdir / 'fig8_confound_vs_insight.png', dpi=150, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# Print summary statistics for the document
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("DEEP ANALYSIS SUMMARY STATISTICS")
print("=" * 80)

print("\n--- Novelty Score Distribution ---")
for n in range(1, 6):
    count = sum(1 for sub in submissions if sub['novelty'] == n)
    examples = [sub['label'] for sub in submissions if sub['novelty'] == n][:5]
    mean_eps = np.mean([sub['epiplexity_per_step'] for sub in submissions if sub['novelty'] == n])
    print(f"  Novelty {n}: {count} submissions, mean eps/step = {mean_eps:.4f}")
    print(f"    Examples: {', '.join(examples)}")

print("\n--- Phase Statistics ---")
for start, end, label, _ in phases:
    end = min(end, len(submissions))
    phase_subs = submissions[start:end]
    mean_eps = np.mean([s['epiplexity_per_step'] for s in phase_subs])
    mean_nov = np.mean([s['novelty'] for s in phase_subs])
    mean_steps = np.mean([s['total_steps'] for s in phase_subs])
    print(f"  {label.replace(chr(10), ' ')}: eps/step={mean_eps:.4f}, "
          f"mean_novelty={mean_nov:.1f}, mean_steps={mean_steps:.0f}")

print("\n--- Autocorrelation (first 5 lags) ---")
for lag in range(6):
    print(f"  Lag {lag}: {autocorr[lag]:.4f}")

print("\n--- Correlation: cum_high_novelty vs eps/step ---")
print(f"  Pearson r = {corr:.4f}")

# Correlation excluding outliers (already computed above)
corr2 = np.corrcoef(x_clean, y_clean)[0, 1]
print(f"  Pearson r (excl outliers) = {corr2:.4f}")

print("\n--- Top 10 Highest Normalized Epiplexity (excl outliers) ---")
sorted_subs = sorted([s for s in submissions if s['label'] not in ['50Bruns', 'llmc', 'ScaleUp1B']], 
                     key=lambda s: s['epiplexity_per_step'], reverse=True)
for s in sorted_subs[:10]:
    print(f"  {s['label']}: {s['epiplexity_per_step']:.4f} (novelty={s['novelty']}, steps={s['total_steps']})")

print("\n--- Top 10 Most Positive Δ(eps/step) ---")
sorted_delta = sorted(submissions[1:], key=lambda s: s['delta_normalized'], reverse=True)
for s in sorted_delta[:10]:
    print(f"  {s['label']}: Δ={s['delta_normalized']:.4f} (novelty={s['novelty']})")

print("\n--- Top 10 Most Negative Δ(eps/step) (biggest efficiency gains) ---")
for s in sorted_delta[-10:]:
    print(f"  {s['label']}: Δ={s['delta_normalized']:.4f} (novelty={s['novelty']})")

print("\nAll figures saved to", outdir)
