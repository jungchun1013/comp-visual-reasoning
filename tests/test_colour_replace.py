"""Unit tests for the colour-subspace replacement (X27; Codex spec
writing/ATTRIBUTE_GEOMETRY_INTERVENTION_SPEC_CODEX_2026-09-17.md §6–§8).  CPU only, no
checkpoint, no dataset.

Run from the repo root:
    PYTHONPATH=src python -m pytest tests/test_colour_replace.py -x -q
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from patch_language_condition import (colour_subspace_folds, replace_colour_coords, matched_rotation,  # noqa: E402
                                      _boot_family, summarise_colour_replace, COLORS, NUM_LAYERS)

torch.manual_seed(0)
D, T, R = 64, 9, 5


def _basis(d=D, r=R, seed=1):
    g = torch.Generator().manual_seed(seed)
    U, _ = torch.linalg.qr(torch.randn(d, r, generator=g, dtype=torch.float64))
    return U


def _tokens(t=T, d=D, seed=2):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(t, d, generator=g, dtype=torch.float64) * torch.linspace(0.5, 3.0, t, dtype=torch.float64)[:, None]


def test_dose_zero_equals_recipient():
    U, x, z = _basis(), _tokens(), _tokens(seed=3)
    xl, theta, inv = replace_colour_coords(x, z, U, 0.0)
    assert torch.allclose(xl, x, atol=1e-9) and not inv.any() and theta.abs().max() < 1e-6


def test_sham_donor_equals_recipient():
    U, x = _basis(), _tokens()
    xl, theta, inv = replace_colour_coords(x, x, U, 1.0)
    assert torch.allclose(xl, x, atol=1e-9) and not inv.any()


def test_norm_preserved_and_colour_coords_match_donor_at_dose_one():
    U, x, z = _basis(), _tokens(), _tokens(seed=3)
    for lam in (0.5, 1.0):
        xl, theta, inv = replace_colour_coords(x, z, U, lam)
        assert not inv.any()
        assert torch.allclose(xl.norm(dim=-1), x.norm(dim=-1), atol=1e-9)
        # the direction outside the colour subspace is that of the recipient
        P = U @ U.T
        res_x = (x / x.norm(dim=-1, keepdim=True)) @ (torch.eye(D, dtype=x.dtype) - P)
        res_l = (xl / xl.norm(dim=-1, keepdim=True)) @ (torch.eye(D, dtype=x.dtype) - P)
        cos = (res_x * res_l).sum(-1) / (res_x.norm(dim=-1) * res_l.norm(dim=-1))
        assert (cos > 1 - 1e-9).all()
    xl, _, _ = replace_colour_coords(x, z, U, 1.0)
    w = z / z.norm(dim=-1, keepdim=True)
    assert torch.allclose((xl / xl.norm(dim=-1, keepdim=True)) @ U, w @ U, atol=1e-9)


def test_unedited_tokens_unchanged_when_embedded():
    U, x_full, z_full = _basis(), _tokens(t=20), _tokens(t=20, seed=3)
    idx = torch.tensor([1, 4, 7])
    xl, _, _ = replace_colour_coords(x_full[idx], z_full[idx], U, 1.0)
    out = x_full.clone()
    out[idx] = xl
    mask = torch.ones(20, dtype=torch.bool)
    mask[idx] = False
    assert torch.equal(out[mask], x_full[mask])


def test_invalid_flags_not_silent():
    U, x, z = _basis(), _tokens(), _tokens(seed=3)
    x[0] = 0.0                                   # zero recipient norm
    z[1] = 0.0                                   # zero donor norm
    _, _, inv = replace_colour_coords(x, z, U, 1.0)
    assert inv[0] and inv[1] and not inv[2:].any()


def test_rotation_matches_angle_and_norm():
    U, x, z = _basis(), _tokens(), _tokens(seed=3)
    xl, theta, _ = replace_colour_coords(x, z, U, 1.0)
    u = x / x.norm(dim=-1, keepdim=True)
    for kw in ({}, {"shared": True}, {"exclude_U": U}):
        xr, inv = matched_rotation(x, theta, 0, **kw)
        assert not inv.any()
        assert torch.allclose(xr.norm(dim=-1), x.norm(dim=-1), atol=1e-9)
        cos = (u * xr).sum(-1) / xr.norm(dim=-1)
        assert torch.allclose(torch.arccos(cos.clamp(-1, 1)), theta, atol=1e-7)
        # same Euclidean edit length as the colour edit
        assert torch.allclose((xr - x).norm(dim=-1), (xl - x).norm(dim=-1), atol=1e-7)


def test_targeted_rotation_tangent_is_outside_colour_span():
    U, x, z = _basis(), _tokens(), _tokens(seed=3)
    xl, theta, _ = replace_colour_coords(x, z, U, 1.0)
    xr, _ = matched_rotation(x, theta, 3, exclude_U=U)
    r = x.norm(dim=-1, keepdim=True)
    u = x / r
    e = (xr / r - torch.cos(theta)[:, None] * u) / torch.sin(theta)[:, None]
    assert (e @ U).abs().max() < 1e-8
    assert ((e * u).sum(-1)).abs().max() < 1e-8


def test_shared_rotation_is_more_coherent_than_independent():
    U, x, z = _basis(), _tokens(t=30), _tokens(t=30, seed=3)
    xl, theta, _ = replace_colour_coords(x, z, U, 1.0)

    def coherence(y):
        d = y - x
        d = d / d.norm(dim=-1, keepdim=True)
        G = d @ d.T
        n = d.shape[0]
        return float((G.sum() - G.diagonal().sum()) / (n * (n - 1)))
    ind = np.mean([coherence(matched_rotation(x, theta, s)[0]) for s in range(5)])
    sh = np.mean([coherence(matched_rotation(x, theta, s, shared=True)[0]) for s in range(5)])
    assert sh > ind + 0.1


def test_seeds_are_reproducible():
    U, x, z = _basis(), _tokens(), _tokens(seed=3)
    _, theta, _ = replace_colour_coords(x, z, U, 1.0)
    a, _ = matched_rotation(x, theta, 7)
    b, _ = matched_rotation(x, theta, 7)
    c, _ = matched_rotation(x, theta, 8)
    assert torch.equal(a, b) and not torch.allclose(a, c)


def test_subspace_folds_rank_orthonormal_and_heldout_exclusion():
    rng = np.random.RandomState(0)
    N, d = 140, 32
    vals = np.array([COLORS[i % len(COLORS)] for i in range(N)])
    centres = {c: rng.randn(d) * 3 for c in COLORS}
    om = np.zeros((N, 2, NUM_LAYERS, d), np.float32)
    for i in range(N):
        om[i, 0, NUM_LAYERS - 1] = centres[vals[i]] + rng.randn(d) * 0.3
    labels = [{"target": {"color": vals[i]}, "pair_index": i} for i in range(N)]
    folds, fold_of = colour_subspace_folds({"obj_mean": om}, labels, n_folds=5)
    for k, f in folds.items():
        U = f["U"]
        assert f["rank"] <= len(COLORS) - 1 and U.shape[1] == f["rank"]
        assert np.allclose(U.T @ U, np.eye(f["rank"]), atol=1e-5)
        assert not (set(f["fit_pair_index"]) & set(f["heldout_pair_index"]))
        assert set(f["heldout_pair_index"]) == {i for i in range(N) if i % 5 == k}
        assert f["heldout_acc"] > 0.9


def test_boot_family_clusters():
    x = np.array([1.0, 1.0, 3.0, 3.0, 5.0, 5.0])
    fam = np.array([0, 0, 1, 1, 2, 2])
    b = _boot_family(x, fam, n=200, seed=1)
    assert b["mean"] == 3.0 and b["n_families"] == 3 and b["lo"] <= 3.0 <= b["hi"]
    b975 = _boot_family(x, fam, n=200, seed=1, level=0.975)
    assert b975["lo"] <= b["lo"] and b975["hi"] >= b["hi"]


def _row(i, cond, variant, role, dose, seed, margin, p_other, correct, own_cos, **extra):
    base = {"i": i, "pair_index": i, "fold": i % 5, "cond": cond, "variant": variant, "role": role, "dose": dose,
            "seed": seed, "correct": "red", "other": "blue", "correct_id": 5, "other_id": 6,
            "margin": margin, "p_correct": 1 - p_other, "p_other": p_other, "argmax": 5 if correct else 6,
            "is_correct": correct, "logits_colours": {}}
    if variant != "clean":
        base.update({"n_tokens": 4, "obj_colour": "red", "own_cos": own_cos, "unq_shape_cos": 0.1,
                     "unq_material_cos": 0.1, "unq_size_cos": 0.1, "obj_mean_disp": 0.5, "edit_len_mean": 0.5,
                     "edit_len_max": 0.7, "patch_coherence": 0.3, "theta_mean": 0.2, "theta_max": 0.3,
                     "u_dist_to_donor": 0.1, "n_invalid": 0})
    base.update(extra)
    return base


def _synthetic_rows(edited_cos=None):
    """Referent edit lowers the margin by 2 (rotations by 0.5); non-referent edit raises
    P(other) but leaves the margin; edited own-cos = clean + lam (donor − clean) unless
    `edited_cos(role, lam, clean, donor)` overrides it."""
    rows = []
    for i in range(12):
        for c in ("c1", "c2"):
            rows.append(_row(i, c, "clean", "none", 0.0, -1, 4.0, 0.05, True, None))
            for role in ("referent", "nonreferent"):
                cl, dn = (0.6, 0.3) if role == "referent" else (0.1, 0.4)
                rows.append(_row(i, c, "sham", role, 1.0, -1, 4.0, 0.05, True, None, sham_equal=True,
                                 clean={"own_cos": cl, "unq_shape_cos": 0.1, "unq_material_cos": 0.1, "unq_size_cos": 0.1},
                                 donor={"own_cos": dn, "unq_shape_cos": 0.1, "unq_material_cos": 0.1, "unq_size_cos": 0.1}))
                for lam in (0.0, 0.5, 1.0):
                    dm = -2.0 * lam if role == "referent" else 0.0
                    po = 0.05 + (0.3 * lam if role == "nonreferent" else 0.0)
                    ce = cl + lam * (dn - cl) if edited_cos is None else edited_cos(role, lam, cl, dn)
                    rows.append(_row(i, c, "edit", role, lam, -1, 4.0 + dm, po, True, ce))
                    if lam == 0:
                        continue
                    for kind in ("rot", "rot_shared", "rot_targeted"):
                        for sd in range(3):
                            rows.append(_row(i, c, kind, role, lam, sd, 4.0 - 0.5 * lam, 0.05, True, cl))
    return rows


def _write_rows(tmp_path, rows):
    with open(tmp_path / "per_image.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_summary_signs_on_synthetic_rows(tmp_path):
    _write_rows(tmp_path, _synthetic_rows())
    out = summarise_colour_replace(tmp_path)
    ref = out["roles"]["referent"]["dose_1"]
    non = out["roles"]["nonreferent"]["dose_1"]
    assert abs(ref["T0_margin_edit_minus_clean"]["mean"] + 2.0) < 1e-9
    assert abs(ref["controls"]["rot"]["T1_margin_edit_minus_control"]["mean"] + 1.5) < 1e-9
    assert ref["controls"]["rot"]["T1_margin_edit_minus_control"]["level"] == 0.975
    assert abs(non["T0_margin_edit_minus_clean"]["mean"]) < 1e-9
    assert abs(non["p_other_edit_minus_clean"]["mean"] - 0.3) < 1e-9
    assert abs(out["role_difference"]["dose_1"]["D_ref_minus_D_nonref"]["mean"] - (-1.5 - 0.5)) < 1e-9
    for role in ("referent", "nonreferent"):
        mp = out["roles"][role]["dose_1"]["manipulation"]
        assert abs(mp["rho"]["mean"] - 1.0) < 1e-9 and mp["frac_pairs_success"] == 1.0 and mp["manipulation_ok"]
        assert mp["frac_pairs_overshoot_past_donor"] == 0.0 and mp["frac_pairs_farther_than_clean"] == 0.0
        mp5 = out["roles"][role]["dose_0.5"]["manipulation"]
        assert abs(mp5["rho"]["mean"] - 0.5) < 1e-9
    dref, dnon = out["decision_fields"]["referent"]["dose_1"], out["decision_fields"]["nonreferent"]["dose_1"]
    assert dref["manipulation_ok"] and dref["H1_T0_margin_hi_below_0"] and dref["H1_T1_margin_vs_rot_hi_below_0"]
    assert dref["primary_endpoint"].startswith("margin")
    # H2 is judged on its own endpoint, P(other colour), not on the margin
    assert dnon["H1_T0_margin_hi_below_0"] is False
    assert dnon["H2_p_other_edit_minus_clean_lo_above_0"] and dnon["H2_p_other_vs_rot_lo_above_0"]
    assert dnon["primary_endpoint"].startswith("P(other")


def test_manipulation_rule_rejects_overshoot(tmp_path):
    """Reviewer counter-example: clean 0.6, donor 0.4 → edited 0.0 gives rho = 3 with the right
    sign, but the edited object is FARTHER from the donor than clean was; must not count as
    success, and must be reported as overshoot."""
    def over(role, lam, cl, dn):
        return cl + 3.0 * lam * (dn - cl)            # rho = 3 at dose 1 (referent: 0.6 → −0.3; donor 0.3)
    _write_rows(tmp_path, _synthetic_rows(over))
    out = summarise_colour_replace(tmp_path)
    mp = out["roles"]["referent"]["dose_1"]["manipulation"]
    assert abs(mp["rho"]["mean"] - 3.0) < 1e-9
    assert mp["frac_pairs_sign_consistent"] == 1.0
    assert mp["frac_pairs_success"] == 0.0 and mp["frac_pairs_overshoot_past_donor"] == 1.0
    assert mp["frac_pairs_farther_than_clean"] == 1.0
    assert not mp["manipulation_ok"]
    assert out["roles"]["referent"]["dose_1"]["sensitivity_all_pairs_success"]["n"] == 0
    # mild overshoot (rho = 1.5: past the donor but still closer than clean) is success, flagged as overshoot
    _write_rows(tmp_path, _synthetic_rows(lambda role, lam, cl, dn: cl + 1.5 * lam * (dn - cl)))
    out = summarise_colour_replace(tmp_path)
    mp = out["roles"]["referent"]["dose_1"]["manipulation"]
    assert mp["frac_pairs_success"] == 1.0 and mp["frac_pairs_overshoot_past_donor"] == 1.0 and mp["manipulation_ok"]


def test_manipulation_rule_is_per_pair_not_image_average(tmp_path):
    """One question direction fails (overshoot) while the other succeeds: image-level averaging
    of cosines would hide it; the pair-level success fraction must show 0.5."""
    rows = _synthetic_rows()
    for r in rows:
        if r["variant"] == "edit" and r["role"] == "referent" and r["cond"] == "c2" and r["dose"] == 1.0:
            r["own_cos"] = 0.6 + 3.0 * (0.3 - 0.6)
    _write_rows(tmp_path, rows)
    out = summarise_colour_replace(tmp_path)
    mp = out["roles"]["referent"]["dose_1"]["manipulation"]
    assert abs(mp["frac_pairs_success"] - 0.5) < 1e-9 and not mp["manipulation_ok"]
    assert out["roles"]["referent"]["dose_1"]["sensitivity_all_pairs_success"]["n"] == 0


def test_summary_tolerates_excluded_pairs(tmp_path):
    """A dropped (image, question) pair (invalid-token rule) leaves the other pair of the image
    in the analysis and does not break the summary."""
    rows = [r for r in _synthetic_rows() if not (r["i"] == 3 and r["cond"] == "c2")]
    _write_rows(tmp_path, rows)
    out = summarise_colour_replace(tmp_path)
    assert out["n_images"] == 12
    assert out["roles"]["referent"]["dose_1"]["manipulation"]["n_pairs"] == 23
