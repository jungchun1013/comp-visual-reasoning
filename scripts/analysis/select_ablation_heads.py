"""G2 head selector for the relational head-ablation experiment (實驗一 step 2).

Reads a headwise activation-patching stats json (schema: top-level `gca_layers`
+ `<group>_<direction>` -> `<category>` -> {sa_mean(12x12), gca_mean(6x16), ...})
and picks the top anchor-binding heads to ablate, plus a disjoint random-head
control of the same composition. Emits three lines:

    ABLATE gca:7:9,gca:5:0,gca:9:3,sa:11:3
    RANDOM gca:1:4,gca:3:11,gca:11:2,sa:4:7
    G2 PASS|FAIL  top=.. median=.. 3xmedian=..

Exit code 0 if the G2 effect-size gate passes (top |mean| > 3x median AND
>= floor), 2 if it fails. The queue reads ABLATE/RANDOM and branches on the code.
gca_mean row i corresponds to layer gca_layers[i].
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--group", default="anchor_swap_noising")
    ap.add_argument("--category", default="anchor_swap")
    ap.add_argument("--n-gca", type=int, default=3)
    ap.add_argument("--n-sa", type=int, default=1)
    ap.add_argument("--floor", type=float, default=0.5,
                    help="G2: top |mean| must be >= this")
    ap.add_argument("--ratio", type=float, default=3.0,
                    help="G2: top |mean| must exceed this x median |mean|")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = json.load(open(args.stats))
    gca_layers = d["gca_layers"]
    sub = d[args.group][args.category]
    sa = np.abs(np.array(sub["sa_mean"]))    # (n_sa_layers, 12) [layer, head]
    gca = np.abs(np.array(sub["gca_mean"]))  # (n_gca_layers, 16) [gca_pos, head]

    heads = []  # (kind, layer, head, |mean|)
    for l in range(sa.shape[0]):
        for h in range(sa.shape[1]):
            heads.append(("sa", l, h, float(sa[l, h])))
    for i in range(gca.shape[0]):
        for h in range(gca.shape[1]):
            heads.append(("gca", gca_layers[i], h, float(gca[i, h])))

    mags = np.array([m for *_, m in heads])
    med = float(np.median(mags))
    top = float(mags.max())

    gca_sorted = sorted((x for x in heads if x[0] == "gca"), key=lambda x: -x[3])
    sa_sorted = sorted((x for x in heads if x[0] == "sa"), key=lambda x: -x[3])
    sel = gca_sorted[:args.n_gca] + sa_sorted[:args.n_sa]
    sel_keys = {(k, l, h) for k, l, h, _ in sel}
    ablate = ",".join(f"{k}:{l}:{h}" for k, l, h, _ in sel)

    rng = random.Random(args.seed)
    gca_pool = [x for x in heads if x[0] == "gca" and (x[0], x[1], x[2]) not in sel_keys]
    sa_pool = [x for x in heads if x[0] == "sa" and (x[0], x[1], x[2]) not in sel_keys]
    rnd = rng.sample(gca_pool, args.n_gca) + rng.sample(sa_pool, args.n_sa)
    random_str = ",".join(f"{k}:{l}:{h}" for k, l, h, _ in rnd)

    passed = (top > args.ratio * med) and (top >= args.floor)
    print(f"ABLATE {ablate}")
    print(f"RANDOM {random_str}")
    print(f"G2 {'PASS' if passed else 'FAIL'}  "
          f"top={top:.4f} median={med:.4f} {args.ratio}xmedian={args.ratio * med:.4f} "
          f"floor={args.floor}")
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
