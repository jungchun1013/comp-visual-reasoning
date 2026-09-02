"""Summary table of the language-condition suite (patch_language_condition/*):
one row per (backbone, queried attribute) run, built from the result JSONs.
No existing aggregator covers these files (aggregate_results.py covers
training runs; probe_table.py covers linear_probe/).

Run from main/:
  PYTHONPATH=src <py> scripts/analysis/language_condition_table.py
Writes outputs/analysis/patch_language_condition/summary_table.{csv,md}
and is imported by aggregate_results.py (language_condition_section_lines).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("outputs/analysis/patch_language_condition")
RUNS = [  # (label, subdir, backbone, queried attribute)
    ("DINOv2 / colour", ".", "DINOv2", "color"),
    ("DINOv2 / shape", "shape", "DINOv2", "shape"),
    ("DINOv2 / material", "material", "DINOv2", "material"),
    ("DINOv2 / size", "size", "DINOv2", "size"),
    ("SigLIP / colour", "siglip", "SigLIP", "color"),
    ("MAE / colour", "mae", "MAE", "color"),
]
COLS = [
    ("sel9", "selection contrast at block 9"),
    ("sel11", "selection contrast at block 11"),
    ("rise8", "queried attr, question − no question, block 8 (referent)"),
    ("rsa_none11", "RSA queried-attr RDM, no question, block 11"),
    ("rsa_ref11", "RSA queried-attr RDM, refer target, block 11"),
    ("rsa_nonref11", "RSA queried-attr RDM, refer distractor, block 11"),
    ("swap_obj9", "answer follows objects' tokens, block 9"),
    ("swap_obj11", "answer follows objects' tokens, block 11"),
    ("swap_bg11", "answer follows background tokens, block 11"),
    ("attn_ref", "decoder attention per referent patch ×1e3"),
    ("attn_nonref", "decoder attention per non-referent patch ×1e3"),
    ("flip5", "flip by queried-attr vector, block 5"),
    ("flip11", "flip by queried-attr vector, block 11"),
    ("probe_ref7", "per-patch referent probe, L7"),
    ("acc", "baseline accuracy (refer target)"),
]


def _load(path):
    return json.load(open(path)) if path.exists() else None


def run_row(label, sub, backbone, queried):
    d = ROOT / sub
    r = {"run": label, "backbone": backbone, "queried": queried}
    a = _load(d / "partA_attr_directions.json")
    if a:
        sel = a["delta"].get(f"ref_target_{queried}_own")
        rise = a["delta"].get(f"refvs0_target_{queried}_own")
        r["sel9"] = sel[9]["mean"] if sel else None
        r["sel11"] = sel[11]["mean"] if sel else None
        r["rise8"] = rise[8]["mean"] if rise else None
    t = _load(d / "partA_rsa_template.json")
    if t:
        key = ("colour" if queried == "color" else queried) + "_template"
        c = t["conditions"]
        r["rsa_none11"] = c["c0"][key][11] if key in c["c0"] else None
        r["rsa_ref11"] = c["c1"][key][11] if key in c["c1"] else None
        r["rsa_nonref11"] = c["c2"][key][11] if key in c["c2"] else None
    sw = _load(d / "readout_swap.json")
    if sw:
        def p(dc, mk, l):
            rr = [x for x in sw["rows"] if x["donor"] == dc and x["mask"] == mk and x["layer"] == l]
            return rr[0]["p_distractor"] if rr else None
        r["swap_obj9"], r["swap_obj11"], r["swap_bg11"] = p("c2", "objects", 9), p("c2", "objects", 11), p("c2", "bg", 11)
    at = _load(d / "readout_attention.json")
    if at:
        m = at["conditions"]["c1"]["mass_per_token"]
        r["attn_ref"], r["attn_nonref"] = m["target"] * 1e3, m["distractor"] * 1e3
        r["acc"] = at["baseline_accuracy"]["c1"]
    iv = _load(d / "intervention_results.json")
    if iv:
        def f(l):
            rr = [x for x in iv["rows"] if x["variant"] == "target_delta_c1" and x["alpha"] == 1.0 and x["layer"] == l]
            return rr[0]["flip_rate"] if rr else None
        r["flip5"], r["flip11"] = f(5), f(11)
    pr = _load(d / "probe_results.json")
    if pr and "L7" in pr:
        r["probe_ref7"] = pr["L7"].get("referent/random_group_image")
    return r


def build_rows():
    return [run_row(*run) for run in RUNS if (ROOT / run[1]).exists()]


SIGNED = {"sel9", "sel11", "rise8"}          # projection differences: signed, one decimal
SCALED = {"attn_ref", "attn_nonref"}          # ×1e3, one decimal


def _fmt(k, v):
    if v is None:
        return "—"
    if k in SIGNED:
        return f"{v:+.1f}"
    if k in SCALED:
        return f"{v:.1f}"
    return f"{v:.2f}"


def language_condition_section_lines():
    rows = build_rows()
    lines = ["## Language-condition suite (patch tokens under referring questions)", "",
             "One row per run (backbone / queried attribute). Selection contrast = target's projection on its own "
             "queried-attribute direction, refer target − refer distractor. RSA = Spearman correlation of the "
             "target's object vector (per-position background vector subtracted) with the queried-attribute model RDM. "
             "'Answer follows X' = proportion of images whose answer becomes the distractor's value after "
             "replacing token group X from the refer-distractor forward pass at that block.", "",
             "| run | " + " | ".join(name for name, _ in COLS) + " |",
             "|" + "---|" * (len(COLS) + 1)]
    for r in rows:
        lines.append("| " + r["run"] + " | " + " | ".join(_fmt(k, r.get(k)) for k, _ in COLS) + " |")
    lines += ["", "Column definitions: " + "; ".join(f"**{k}** = {d}" for k, d in COLS) + ".", ""]
    return lines, rows


def main():
    lines, rows = language_condition_section_lines()
    (ROOT / "summary_table.md").write_text("\n".join(lines))
    with open(ROOT / "summary_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "backbone", "queried"] + [k for k, _ in COLS])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    print("\n".join(lines))
    print(f"Saved: {ROOT / 'summary_table.md'}, {ROOT / 'summary_table.csv'}")


if __name__ == "__main__":
    main()
