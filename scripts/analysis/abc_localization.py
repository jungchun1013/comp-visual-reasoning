#!/usr/bin/env python
"""A/B/C perturbation localization contrast: CA heads vs SA heads (E9, v2 §A4).

Tests the claim: text-side perturbations (A = described attribute, B = queried
attribute) load on cross-attention (GCA) heads, while image-side perturbations
(C = queried attribute swapped in the rendered image) load on self-attention
heads.

Pure aggregation over EXISTING activation-patching stats (read-only):
  outputs/analysis/activation_patching/<model>/headwise_by_type_stats.json
      fine_attribute_denoising          -> A  (described-attr text swap)
      fine_attribute_query_denoising    -> B  (queried-attr text swap)
  outputs/analysis/activation_patching/<model>/visual_<attr>_stats.json
      visual_<attr>_denoising           -> C  (queried-attr image swap)

Each record: sa_mean (n_layers x n_sa_heads), gca_mean (n_gca x n_gca_heads),
per-head mean delta-logit for the correct answer under denoising (restore clean
head activation into corrupted run; positive = head carries the recovery).

Metrics per (perturbation, attribute):
  - mass share: sum(|gca|) / (sum(|sa|) + sum(|gca|)), per-head normalized
  - top-head: location + value of the strongest CA and SA head
  - top10 share: fraction of the 10 strongest heads (pooled) that are CA

Outputs (new dir, never touches inputs):
  outputs/analysis/abc_localization/<model>/abc_contrast.json + abc_contrast.md

Usage (from main/):
  <interpreter> scripts/analysis/abc_localization.py \
      [--model clevr_dinov2_decoder1l_scratch]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ATTRS = ["color", "material", "size", "shape"]


def head_records(section: dict, gca_layers: list[int]):
    """Yield (kind, layer, head, value) for every head in one stats record."""
    sa = np.asarray(section["sa_mean"])
    gca = np.asarray(section["gca_mean"])
    recs = []
    for l in range(sa.shape[0]):
        for h in range(sa.shape[1]):
            recs.append(("SA", l, h, float(sa[l, h])))
    for i, l in enumerate(gca_layers):
        for h in range(gca.shape[1]):
            recs.append(("CA", l, h, float(gca[i, h])))
    return recs


def summarize(section: dict, gca_layers: list[int]) -> dict:
    recs = head_records(section, gca_layers)
    sa_abs = np.array([abs(v) for k, _, _, v in recs if k == "SA"])
    ca_abs = np.array([abs(v) for k, _, _, v in recs if k == "CA"])
    # per-head means so head-count imbalance (144 SA vs 96 CA) doesn't bias the share
    sa_m, ca_m = sa_abs.mean(), ca_abs.mean()
    top = sorted(recs, key=lambda r: abs(r[3]), reverse=True)[:10]
    top_ca = max((r for r in recs if r[0] == "CA"), key=lambda r: abs(r[3]))
    top_sa = max((r for r in recs if r[0] == "SA"), key=lambda r: abs(r[3]))
    return {
        "n_samples": section.get("n"),
        "ca_mean_abs": float(ca_m),
        "sa_mean_abs": float(sa_m),
        "ca_share_perhead": float(ca_m / (ca_m + sa_m)),
        "top_ca_head": {"layer": top_ca[1], "head": top_ca[2], "delta": top_ca[3]},
        "top_sa_head": {"layer": top_sa[1], "head": top_sa[2], "delta": top_sa[3]},
        "top10_ca_count": sum(1 for r in top if r[0] == "CA"),
        "top10": [{"kind": k, "layer": l, "head": h, "delta": v}
                  for k, l, h, v in top],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="clevr_dinov2_decoder1l_scratch")
    ap.add_argument("--patching-root", default="outputs/analysis/activation_patching")
    ap.add_argument("--output-root", default="outputs/analysis/abc_localization")
    args = ap.parse_args()

    src = Path(args.patching_root) / args.model
    out_dir = Path(args.output_root) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    headwise = json.loads((src / "headwise_by_type_stats.json").read_text())
    gca_layers = headwise["gca_layers"]

    result = {"model": args.model, "gca_layers": gca_layers, "perturbations": {}}
    sections = {
        "A_described_attr_text": {
            a: headwise["fine_attribute_denoising"][a] for a in ATTRS},
        "B_queried_attr_text": {
            a: headwise["fine_attribute_query_denoising"][f"what_{a}"] for a in ATTRS},
    }
    c_sections = {}
    for a in ATTRS:
        p = src / f"visual_{a}_stats.json"
        if p.exists():
            v = json.loads(p.read_text())
            c_sections[a] = v[f"visual_{a}_denoising"][a]
    if c_sections:
        sections["C_queried_attr_image"] = c_sections

    for pert, per_attr in sections.items():
        result["perturbations"][pert] = {
            a: summarize(sec, gca_layers) for a, sec in per_attr.items()}

    (out_dir / "abc_contrast.json").write_text(json.dumps(result, indent=2))

    lines = [f"# A/B/C localization contrast — {args.model}", "",
             "CA share = per-head mean |delta-logit| of CA heads / (CA + SA), "
             "denoising direction. High = perturbation recovery routes through "
             "cross-attention; low = through self-attention.", "",
             "| perturbation | attr | CA share | top CA head (delta) | top SA head (delta) | CA in top-10 |",
             "|---|---|---|---|---|---|"]
    for pert, per_attr in result["perturbations"].items():
        for a, s in per_attr.items():
            tc, ts = s["top_ca_head"], s["top_sa_head"]
            lines.append(
                f"| {pert} | {a} | {s['ca_share_perhead']:.3f} "
                f"| L{tc['layer']}H{tc['head']} ({tc['delta']:+.3f}) "
                f"| L{ts['layer']}H{ts['head']} ({ts['delta']:+.3f}) "
                f"| {s['top10_ca_count']}/10 |")
    (out_dir / "abc_contrast.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_dir}/abc_contrast.{{json,md}}")
    for line in lines[5:]:
        print(line)


if __name__ == "__main__":
    main()
