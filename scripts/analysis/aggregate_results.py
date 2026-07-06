#!/usr/bin/env python
"""Aggregate all experiment results into docs/results_tables.md.

Read-only over outputs/. Walks:
  - outputs/model/*/          final val acc (train_log.jsonl, else text logs; conflicts shown)
  - outputs/analysis/generalization/*.json   per-question-type breakdowns
  - outputs/analysis/cogent_sample_efficiency/sample_efficiency.json
  - outputs/analysis/*/       artifact inventory (presence + mtime)

Usage (from main/):
  PYTHONPATH=src <interpreter> scripts/analysis/aggregate_results.py [--out docs/results_tables.md]

Log-format facts (see docs/paper_artifacts.md): train_log.jsonl rows have per-epoch
val_acc; text logs print per-epoch "Val acc: X" and a final "Done. Best val acc: X".
Paper convention = final-epoch accuracy. When multiple sources disagree for one run
dir (known case: clevr_dinov2_concat_decoder1l_scratch_s42 stdout.log is a foreign
run's log), all values are listed and the conflict is flagged rather than resolved
silently — checkpoint-stored val_acc is authoritative (--ckpt-meta reads it).
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

VAL_ACC_RE = re.compile(r"Val acc: ([0-9.]+)")

MODEL_ROOT = Path("outputs/model")
ANALYSIS_ROOT = Path("outputs/analysis")

# Run dirs whose true training log lives under a different sibling name.
# clevr_dinov2_concat_decoder1l_scratch_s42/stdout.log belongs to a foreign run
# (dir contamination — see docs/paper_artifacts.md §8.1); the real log is the
# top-level file below, which agrees with the checkpoint-stored val_acc 0.9237.
KNOWN_LOG_ALIASES = {
    "clevr_dinov2_concat_decoder1l_scratch_s42":
        "clevr_dinov2_concat_decoder_scratch_s42.log",
}


def final_acc_from_jsonl(path: Path):
    last = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    if not last:
        return None
    try:
        rec = json.loads(last)
    except json.JSONDecodeError:
        return None
    if "val_acc" in rec:
        return rec.get("epoch"), float(rec["val_acc"])
    return None


def final_acc_from_textlog(path: Path):
    """Last per-epoch 'Val acc:' line (final-epoch accuracy, paper convention)."""
    vals = []
    try:
        with open(path, errors="replace") as f:
            for line in f:
                m = VAL_ACC_RE.search(line)
                if m:
                    vals.append(float(m.group(1)))
    except OSError:
        return None
    return (None, vals[-1]) if vals else None


def collect_model_accs():
    """Per run dir: list of (source, epoch, acc)."""
    rows = {}
    for d in sorted(MODEL_ROOT.iterdir()):
        if not d.is_dir():
            continue
        found = []
        jl = d / "train_log.jsonl"
        if jl.exists():
            r = final_acc_from_jsonl(jl)
            if r:
                found.append(("train_log.jsonl", r[0], r[1]))
        cands = [MODEL_ROOT / f"{d.name}.log", MODEL_ROOT / f"{d.name}_train.log"]
        if d.name in KNOWN_LOG_ALIASES:
            cands.insert(0, MODEL_ROOT / KNOWN_LOG_ALIASES[d.name])
        cands += sorted(d.glob("*.log"))
        for cand in cands:
            if cand.exists():
                r = final_acc_from_textlog(cand)
                if r:
                    found.append((cand.name, r[0], r[1]))
        rows[d.name] = found
    return rows


def collect_ckpt_meta(names):
    """Authoritative val_acc from checkpoint metadata (slow: torch.load per file)."""
    import torch  # deferred: only needed with --ckpt-meta
    meta = {}
    for name in names:
        for ck in ("best.pt", "last.pt"):
            p = MODEL_ROOT / name / ck
            if p.exists():
                try:
                    d = torch.load(p, map_location="cpu", weights_only=False)
                    meta[name] = (ck, d.get("epoch"), d.get("val_acc", d.get("best_acc")))
                except Exception as e:  # noqa: BLE001 — inventory must not die on one bad file
                    meta[name] = (ck, None, f"load error: {e}")
                break
    return meta


def collect_generalization():
    rows = {}
    gen = ANALYSIS_ROOT / "generalization"
    if not gen.exists():
        return rows
    for p in sorted(gen.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        std = d.get("clevr_standard")
        row = {"file": p.name, "epoch": d.get("epoch")}
        if std:
            row["overall"] = std.get("accuracy")
            row["breakdown"] = {k: v.get("accuracy")
                                for k, v in (std.get("breakdown") or {}).items()}
        for extra in ("clevr_humans", "cogent", "closure"):
            if extra in d:
                row[extra] = d[extra]
        rows[d.get("model", p.stem)] = row
    return rows


def collect_cogent():
    p = ANALYSIS_ROOT / "cogent_sample_efficiency" / "sample_efficiency.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {"before": d.get("before"),
            "final_per_n": {k: v[-1] for k, v in d.get("runs", {}).items() if v}}


def artifact_inventory():
    rows = []
    for d in sorted(ANALYSIS_ROOT.iterdir()):
        if d.is_dir():
            files = list(d.iterdir())
            mtime = max((f.stat().st_mtime for f in files), default=d.stat().st_mtime)
            rows.append((d.name, len(files),
                         datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")))
    return rows


def fmt(x):
    if isinstance(x, float):
        return f"{x:.4f}"
    return "—" if x is None else str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/results_tables.md")
    ap.add_argument("--ckpt-meta", action="store_true",
                    help="also read authoritative val_acc from checkpoint files (slow)")
    args = ap.parse_args()

    accs = collect_model_accs()
    ckpt = collect_ckpt_meta(accs.keys()) if args.ckpt_meta else {}
    gen = collect_generalization()
    cogent = collect_cogent()
    inventory = artifact_inventory()

    lines = ["# Results tables (generated — do not hand-edit)",
             "",
             f"Generated by `scripts/analysis/aggregate_results.py` on "
             f"{datetime.now():%Y-%m-%d %H:%M}. Numbers are final-epoch val acc "
             "(paper convention; see docs/paper_artifacts.md).",
             "",
             "## Final val accuracy per run",
             "",
             "| run dir | acc | source | conflicts |"]
    lines.append("|---|---|---|---|")
    for name, found in accs.items():
        if not found and name not in ckpt:
            lines.append(f"| {name} | — | no parsable log | |")
            continue
        vals = {round(v, 4) for _, _, v in found}
        best_src, _, best_val = found[0] if found else ("", None, None)
        conflict = ""
        if len(vals) > 1:
            conflict = "⚠ " + "; ".join(f"{s}={v:.4f}" for s, _, v in found)
        cell = f"{best_val:.4f}" if best_val is not None else "—"
        if name in ckpt:
            ck, ep, v = ckpt[name]
            cell = fmt(v)
            best_src = f"{ck} (authoritative)"
            if found and isinstance(v, float) and abs(found[0][2] - v) > 5e-4:
                conflict = "⚠ logs: " + "; ".join(f"{s}={vv:.4f}" for s, _, vv in found)
        lines.append(f"| {name} | {cell} | {best_src} | {conflict} |")

    if gen:
        qtypes = sorted({q for r in gen.values() for q in (r.get("breakdown") or {})})
        lines += ["", "## Per-question-type breakdown (eval_generalization.py)", "",
                  "| model | overall | " + " | ".join(qtypes) + " |",
                  "|---|---|" + "---|" * len(qtypes)]
        for name, r in sorted(gen.items()):
            bd = r.get("breakdown") or {}
            lines.append(f"| {name} | {fmt(r.get('overall'))} | "
                         + " | ".join(fmt(bd.get(q)) for q in qtypes) + " |")

    if cogent:
        lines += ["", "## CoGenT sample efficiency", "",
                  f"Zero-shot (before ft): valA {fmt(cogent['before'].get('valA'))}, "
                  f"valB {fmt(cogent['before'].get('valB'))}",
                  "", "| n finetune samples | valA | valB |", "|---|---|---|"]
        for n, rec in sorted(cogent["final_per_n"].items(), key=lambda kv: int(kv[0])):
            lines.append(f"| {n} | {fmt(rec.get('valA'))} | {fmt(rec.get('valB'))} |")

    lines += ["", "## Analysis artifact inventory", "",
              "| outputs/analysis/ dir | files | last modified |", "|---|---|---|"]
    for name, n, mtime in inventory:
        lines.append(f"| {name} | {n} | {mtime} |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out} ({len(accs)} model dirs, {len(gen)} generalization files)")


if __name__ == "__main__":
    main()
