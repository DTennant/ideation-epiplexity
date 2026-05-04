# Ideation ≠ Optimization: Measuring Genuine Discovery in Self-Improving Agents via Epiplexity

## Core Question

When a self-improving agent claims to have "improved" its model, did it actually **discover a new structural insight** (ideation), or did it just **tune parameters within a known framework** (optimization)?

We propose using **epiplexity** — the amount of learnable structure a computationally bounded observer extracts from data — to distinguish these two modes.

## Background

### The Problem with Current RSI Benchmarks

Current benchmarks for self-improving AI systems (SWE-bench, MATH, etc.) measure **performance** — did the score go up? But a score increase of 3% tells you nothing about *how* it was achieved:

- **Optimization**: Better hyperparameters, more training, prompt engineering. The model sees the data through the same structural lens. → *No new insight.*
- **Ideation**: A genuinely new way to process the data — discovering skip connections, attention, a new loss function, a new representation. The model can now extract structure it previously couldn't. → *New insight.*

The history of deep learning shows this distinction matters enormously. The jump from plain CNNs to ResNets, or from RNNs to Transformers, didn't just improve numbers — it unlocked entirely new classes of learnable structure.

### Epiplexity (Finzi et al., 2026)

**Epiplexity** $S_{\mathcal{V}}(X)$ measures the total learnable structural information that a model class $\mathcal{V}$ can extract from data $X$. Operationally, it is the area between the training loss curve and the final converged loss:

$$S_{\mathcal{V}}(X) = \int_0^\infty \left[ L_{\mathcal{V}}(t, X) - L_{\mathcal{V}}(\infty, X) \right] dt$$

Key properties:
- Unlike Shannon entropy, epiplexity is **observer-dependent**: the same data has different epiplexity for different model classes
- It separates **structure** (what can be learned) from **randomness** (what cannot)
- It can be **created by computation** — deterministic transformations can increase epiplexity for bounded observers

See: [From Entropy to Epiplexity](https://arxiv.org/abs/2601.03220) (Finzi, Qiu, Jiang, Izmailov, Kolter, Wilson, 2026)

### Related: V-information (Xu et al., 2020)

The precursor framework. Defines *usable information* under computational constraints, showing that information can be created through computation (violating the data processing inequality). Epiplexity builds on this by decomposing information into structural vs. random components.

See: [A Theory of Usable Information Under Computational Constraints](https://arxiv.org/abs/2002.10689) (Xu et al., ICLR 2020)

## Our Framework

### Idea Importance

Given data $X$ and two model classes $\mathcal{V}_{old}$, $\mathcal{V}_{new}$ (where the difference between them represents an "idea" — a structural design choice), the **importance of the idea** is:

$$I(\mathcal{V}_{old} \to \mathcal{V}_{new};\; X) = S_{\mathcal{V}_{new}}(X) - S_{\mathcal{V}_{old}}(X)$$

- $I \gg 0$: The idea unlocks significant new learnable structure → **genuine ideation**
- $I \approx 0$: The idea doesn't change what structure can be extracted → **mere optimization**

### Detecting Ideation in Self-Improving Agents

Setup: An agent iteratively proposes modifications to a model architecture/algorithm, trained on the **same fixed data**.

At each iteration $i$, the agent produces a new model class $\mathcal{V}_i$. We track two quantities:

1. **Performance**: task-specific metric (accuracy, loss, etc.)
2. **Epiplexity**: $S_{\mathcal{V}_i}(X)$

Four possible outcomes at each step:

| Epiplexity $\Delta S$ | Performance $\Delta P$ | Interpretation |
|---|---|---|
| ↑ large | ↑ | **Ideation** — found new structure, and it helps |
| ≈ 0 | ↑ small | **Optimization** — squeezed more from known structure |
| ↑ large | ≈ 0 or ↓ | **Discovery without immediate payoff** — found new structure, not yet useful for this task (but may transfer to others) |
| ≈ 0 | ≈ 0 | **Stagnation** |

The key insight: **performance gains without epiplexity gains are optimization, not ideation.** A system that only produces the second row is not doing genuine ideation, regardless of how many points it gains.

### Idea Synergy

Ideas can be synergistic: their combined value exceeds the sum of their individual values.

Given a base model class $\mathcal{V}_0$ and two ideas $A$, $B$ that produce $\mathcal{V}_A$, $\mathcal{V}_B$, $\mathcal{V}_{AB}$:

$$\text{Synergy}(A, B; X) = S_{\mathcal{V}_{AB}}(X) - S_{\mathcal{V}_A}(X) - S_{\mathcal{V}_B}(X) + S_{\mathcal{V}_0}(X)$$

- Positive synergy: ideas unlock structure *together* that neither unlocks alone (e.g., attention + positional encoding)
- Zero synergy: ideas are independent
- Negative synergy: ideas interfere

## Experimental Design

### Data: Sequences with Hidden Structure

We use synthetic sequences that contain structure requiring specific "insights" to discover. The agent doesn't know the generating rule — it must find the right inductive bias.

**Example 1: XOR-delayed sequence**

$$x_t = x_{t-1} \oplus x_{t-k}$$

Requires two insights: (1) look back $k$ steps, (2) compute XOR (nonlinearity). Neither alone suffices.

**Example 2: Modular arithmetic with distractors**

$$x_t = f(x_{t-3}, x_{t-7}) \mod p, \quad \text{interleaved with noise dimensions}$$

Requires: (1) identify which dimensions are signal vs. noise, (2) discover the modular structure.

**Example 3: Cellular automaton (e.g., Rule 110)**

Deterministic, but produces emergent structures (gliders, oscillators). A model that discovers "objects" as a concept extracts far more structure than one that models pixels independently.

### Protocol

1. **Generate data** from a fixed rule (agent doesn't see the rule)
2. **Self-improving agent** proposes model modifications iteratively:
   - Start with a baseline architecture
   - Each round: agent sees current performance, proposes a change
   - Change is implemented, model retrained from scratch
3. **Measure at each iteration**:
   - Performance (test loss)
   - Epiplexity (loss curve area)
4. **Compare**: Do epiplexity jumps correspond to qualitatively different architectures? Do performance-only gains correspond to parameter/hyperparameter tuning?

### Baselines

- **Random search** in architecture space (non-agent)
- **Grid search** over hyperparameters only (pure optimization)
- **Self-improving agent** with LLM proposing architecture changes

## What This Is Not

This is **not RSI** — the self-improving agent is not recursively improving itself. It's improving a *separate* model on a *fixed* task. The "recursive" and "self" parts of RSI are much harder problems.

What this *is*: a measurement framework that can tell you whether a system is doing **genuine ideation** (discovering new structural insights) or **mere optimization** (tuning within a known paradigm). If we can't even measure the difference, we can't build systems that do the former.

## Repo Structure

```
├── README.md                 # This file
├── data/                     # Data generation scripts
│   ├── xor_delayed.py        # XOR-delayed sequences
│   ├── modular_arith.py      # Modular arithmetic with distractors
│   └── cellular_automata.py  # Rule 110 / Game of Life
├── models/                   # Model definitions
│   └── baselines.py          # Baseline architectures (MLP, RNN, etc.)
├── epiplexity/               # Epiplexity measurement
│   └── measure.py            # Loss curve integration
├── agent/                    # Self-improving agent
│   └── architect.py          # LLM-based architecture proposer
├── experiments/              # Experiment scripts
│   └── run_selfimprove.py    # Main experiment loop
└── analysis/                 # Analysis and plotting
    └── plot_ideation.py      # Epiplexity vs performance plots
```

## References

- Finzi, M., Qiu, S., Jiang, Y., Izmailov, P., Kolter, J.Z., & Wilson, A.G. (2026). *From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence.* [arXiv:2601.03220](https://arxiv.org/abs/2601.03220)
- Xu, Y., Zhao, S., Song, J., Stewart, R., & Ermon, S. (2020). *A Theory of Usable Information Under Computational Constraints.* ICLR 2020. [arXiv:2002.10689](https://arxiv.org/abs/2002.10689)
- Rosas, F.E., Mediano, P.A.M., et al. (2020). *Reconciling emergences: An information-theoretic approach to identify causal emergence in multivariate data.* PLOS Computational Biology.

## License

MIT
