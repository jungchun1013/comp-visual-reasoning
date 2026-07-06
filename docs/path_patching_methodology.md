# Path Patching for Circuit Discovery in SteerViT

## Motivation

We have component-level evidence (head patching) showing *which* heads matter, and geometric evidence (RSA, linear probes) showing *what* information is encoded. But we don't know *how* information flows between components to produce the answer.

Path patching closes this gap: it discovers the computational graph (circuit) that the model uses, without pre-assuming what each component does. We find the circuit first, then interpret.

## Model Structure

Each ViT block has up to 3 components writing to the residual stream:

```
Block L_i:
  residual → GCA(residual, text)  → residual'        [only at GCA layers: 1,3,5,7,9,11]
  residual' → SA(LN(residual'))   → residual''        [self-attention]
  residual'' → MLP(LN(residual'')) → residual'''      [feed-forward]
```

Total components: 6 GCA + 12 SA + 12 MLP = 30 nodes.
Each node reads from and writes to a shared residual stream.

## Method Overview

### Corruption Setup

All experiments use **text corruption**: change a described attribute in the question (e.g., "red" → "green"). This targets the text→vision→answer pathway.

Clean run: model processes (image, clean_question) → correct_answer
Corrupt run: model processes (image, corrupt_question) → wrong_answer

Metric: Δlogit = logit(correct_token | intervened) − logit(correct_token | clean)

Negative Δlogit = intervention hurt performance = component was important.

### Step 1: Node Importance (Activation Patching)

**Goal**: Which components causally affect the answer?

**Method**: For each component C:
1. Run clean forward, record C's output contribution
2. Run corrupt forward, record C's corrupt contribution  
3. Run clean forward again, but replace C's contribution with the corrupt version
4. Measure Δlogit

**Output**: Importance score per component. Ranked list of which GCA/SA/MLP nodes matter.

**What this tells us**: The "what" — which components are necessary for correct answering. But not how they interact.

### Step 2: Edge Discovery (Path Patching)

**Goal**: For each important node (sender), which downstream nodes receive its information?

**Method**: For a sender S and each downstream receiver R:
1. Corrupt S's output (add corruption diff)
2. Freeze R's output to its clean value (undo corruption at R)
3. Measure Δlogit

**Interpretation**:
- If freezing R restores the logit → S's corruption was **mediated through R**
- If freezing R doesn't help → S's effect goes through other paths, not R

**Output**: Edge weight matrix: sender × receiver. Shows information flow.

### Step 3: Circuit Extraction

**Goal**: Find the minimal subgraph (circuit) that explains model behavior.

**Method**: 
1. Start with all edges from Step 2
2. Threshold: keep only edges with |effect| > τ
3. Verify: run the model with only circuit components active (ablate everything else)
4. Check: circuit-only performance ≈ full model performance

**Output**: A directed graph of components and their connections.

### Step 4: Functional Interpretation

**Goal**: Name each component by what it does (discovered, not assumed).

**Method**: For each node in the circuit:
1. Analyze what information it reads (probe its input)
2. Analyze what information it writes (probe its output)
3. Check consistency across different inputs

**Naming convention**: Name by function, e.g.:
- "attribute reader" (a GCA head that extracts queried attribute from text)
- "object selector" (an SA head that routes attention to the target object)
- "answer writer" (an MLP that maps to answer logits)

## Why Not Post-Hoc

Post-hoc approach: hypothesize "L7 GCA does binding" → test if L7 GCA matters for binding.
Problem: confirmation bias. You find what you're looking for.

Discovery approach: find which components matter → trace how they connect → interpret what each does.
Advantage: you might discover unexpected circuits (e.g., MLP layers doing something crucial that you wouldn't have hypothesized).

## Relation to Existing Evidence

| Existing | What it shows | What it doesn't show |
|----------|--------------|---------------------|
| Head patching (CoGenT) | Which SA/GCA heads affect answer | How heads interact |
| RSA | Binding/grounding geometry emerges at L5-L7 | Whether model reads that geometry |
| Linear probe | Answer info decodable from L11 | Whether model uses that direction |

Path patching adds: **the computational graph** connecting these observations.

Expected outcome: the circuit should explain *why* binding geometry emerges where it does — because specific GCA heads write binding info, specific SA heads route it, and specific MLPs read it for the answer.

## Experimental Protocol

### Samples
- Corruption type: `fine_attribute/color` (strongest, cleanest signal)
- N = 50 per step (each sample = one (image, clean_q, corrupt_q, answer) tuple)
- CLEVR val set

### Step 1 Parameters
- Components: 30 (6 GCA + 12 SA + 12 MLP)
- Runtime: ~30 min (50 samples × 30 forward passes)

### Step 2 Parameters
- Senders: top-5 from Step 1
- Receivers per sender: all downstream components
- Runtime: ~2 hours (5 senders × ~20 receivers × 50 samples)

### Step 3 Parameters
- Threshold τ: chosen so circuit has 5-10 nodes
- Verification: circuit-only model accuracy on 500 samples

## Output Structure

```
outputs/analysis/path_patching/<model_name>/
├── step1_node_importance.json       # {component: {mean, std, n}}
├── step1_node_importance.png        # bar chart
├── step2_edges.json                 # {sender: {receiver: {mean, std, n}}}
├── step2_edges_<sender>.png         # per-sender edge bar charts
├── step3_circuit.json               # minimal circuit graph
└── step3_circuit_verification.json  # circuit-only accuracy
```
